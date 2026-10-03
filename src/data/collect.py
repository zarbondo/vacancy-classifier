# сбор вакансий с hh.ru: python main.py collect
#
# api.hh.ru с мая 2026 отдаёт 403 любому программному клиенту, поэтому берём данные
# со страницы поиска. Парсить вёрстку не нужно: hh кладёт в <template id="HH-Lux-InitialState">
# весь ответ в JSON — там и текст, и оба наших таргета.
#
# Ходим по парам (роль, опыт): hh отдаёт максимум 2000 вакансий на запрос, а разбивка
# по опыту даёт 4 независимых среза и заодно выравнивает редкие грейды.
import html as H
import json
import re
import time
from pathlib import Path

import polars as pl
import requests

SEARCH = "https://hh.ru/search/vacancy"
STATE = re.compile(r'<template[^>]*id="HH-Lux-InitialState"[^>]*>(.*?)</template>', re.S)
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9",
}
EXPERIENCE = ["noExperience", "between1And3", "between3And6", "moreThan6"]


def get_state(params, session, retries=3):
    for attempt in range(retries):
        r = session.get(SEARCH, params=params, headers=HEADERS, timeout=30)
        if r.status_code in (403, 429):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 451:
            raise RuntimeError("hh.ru отдаёт 451 — нужен российский IP, из Colab и VPN не выйдет")
        r.raise_for_status()
        m = STATE.search(r.text)
        if not m:
            raise RuntimeError("на странице нет HH-Lux-InitialState — вёрстка hh поменялась")
        return json.loads(H.unescape(m.group(1)))
    raise RuntimeError("hh.ru блокирует запросы — увеличь collect.pause")


def role_names(state):
    """Справочник ролей лежит в том же состоянии страницы, зашивать названия не нужно."""
    out = {}
    for cat in state.get("professionalRoleTree", {}).get("items", []):
        for it in cat.get("items", []):
            out[str(it["id"])] = it["text"]
    return out


def parse(v, names):
    sn = v.get("snippet") or {}
    # req — требования, resp — обязанности; cond это зарплата и график, для задачи шум.
    # tags не берём намеренно: там лежит строка "Опыт 3-6 лет", то есть сам таргет
    text = " ".join(filter(None, [v.get("name"), sn.get("req"), sn.get("resp")]))
    ids = [str(i) for block in (v.get("professionalRoleIds") or [])
           for i in (block.get("professionalRoleId") or [])]
    return {
        "id": int(v["vacancyId"]),
        "text": text,
        "experience": v.get("workExperience"),
        "role_id": ids[0] if ids else None,
        "role_name": names.get(ids[0]) if ids else None,
        "area": (v.get("area") or {}).get("name"),
        "employer": (v.get("company") or {}).get("visibleName"),
    }


def collect(cfg):
    c = cfg["collect"]
    session = requests.Session()
    rows, seen, names = [], set(), {}

    for role in c["roles"]:
        for exp in EXPERIENCE:
            for page in range(c["max_pages"]):
                params = {"professional_role": role, "area": c["area"], "experience": exp,
                          "items_on_page": c["per_page"], "page": page,
                          "enable_snippets": "true", "search_period": c["period"]}
                st = get_state(params, session)
                names = names or role_names(st)
                vacancies = st.get("vacancySearchResult", {}).get("vacancies", [])
                if not vacancies:
                    break
                for v in vacancies:
                    if v["vacancyId"] not in seen:
                        seen.add(v["vacancyId"])
                        rows.append(parse(v, names))
                time.sleep(c["pause"])
            print("роль %s, опыт %-13s всего собрано %d" % (role, exp, len(rows)))

    df = pl.DataFrame(rows).drop_nulls(["experience", "role_name"])
    out = Path(c["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print("\nсохранено %d вакансий в %s" % (df.height, out))
    print(df.group_by("experience").agg(pl.len().alias("n")).sort("n", descending=True))
    return df
