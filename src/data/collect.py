# сбор вакансий через открытое API hh.ru: python main.py collect
# полное описание вакансии лежит за отдельным запросом на каждую (это ~30k запросов),
# поэтому берём name + snippet из выдачи поиска — текста хватает, а сбор занимает минуты
import time
from pathlib import Path

import polars as pl
import requests

API = "https://api.hh.ru/vacancies"
UA = {"User-Agent": "vacancy-classifier/1.0 (github.com/zarbondo)"}


def fetch_page(params, retries=3):
    for attempt in range(retries):
        r = requests.get(API, params=params, headers=UA, timeout=20)
        if r.status_code == 429:  # слишком часто — подождём и повторим
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("hh.ru отвечает 429, слишком частые запросы — увеличь collect.pause")


def parse(item):
    sn = item.get("snippet") or {}
    text = " ".join(filter(None, [
        item.get("name"),
        sn.get("requirement"),
        sn.get("responsibility"),
    ]))
    roles = item.get("professional_roles") or [{}]
    exp = (item.get("experience") or {}).get("id")
    return {
        "id": int(item["id"]),
        "text": text.replace("<highlighttext>", "").replace("</highlighttext>", ""),
        "experience": exp,
        "role_id": roles[0].get("id"),
        "role_name": roles[0].get("name"),
        "area": (item.get("area") or {}).get("name"),
        "employer": (item.get("employer") or {}).get("name"),
    }


def collect(cfg):
    c = cfg["collect"]
    rows, seen = [], set()
    for role in c["roles"]:
        for page in range(c["max_pages"]):
            params = {"professional_role": role, "area": c["area"], "period": c["period"],
                      "per_page": c["per_page"], "page": page}
            data = fetch_page(params)
            items = data.get("items", [])
            if not items:
                break
            for it in items:
                if it["id"] not in seen:
                    seen.add(it["id"])
                    rows.append(parse(it))
            time.sleep(c["pause"])
            if page + 1 >= data.get("pages", 0):
                break
        print("роль %s: всего собрано %d" % (role, len(rows)))

    df = pl.DataFrame(rows).drop_nulls(["experience", "role_id"])
    out = Path(c["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print("сохранено %d вакансий в %s" % (df.height, out))
    return df
