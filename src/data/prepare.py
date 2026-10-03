# чистка текста, разметка и сплит.
# разметку не делал руками: грейд берётся из поля experience, направление — из professional_roles,
# то есть таргеты проставил сам hh.ru
import re

import numpy as np
import polars as pl

# порядок важен: это упорядоченные грейды, а не просто классы
GRADES = ["noExperience", "between1And3", "between3And6", "moreThan6"]
GRADE_RU = {"noExperience": "без опыта", "between1And3": "1–3 года",
            "between3And6": "3–6 лет", "moreThan6": "более 6 лет"}

TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")


def clean(text):
    return SPACE.sub(" ", TAG.sub(" ", text)).strip()


def prepare(df, cfg):
    d = cfg["data"]
    df = df.with_columns(pl.col("text").map_elements(clean, return_dtype=pl.Utf8))
    df = df.filter(pl.col("text").str.len_chars() >= d["min_text_len"])
    df = df.filter(pl.col("experience").is_in(GRADES))
    df = df.unique(subset=["text"])  # одна и та же вакансия часто висит в нескольких ролях

    # редкие роли сливаем в "прочее", иначе в тесте будут классы из двух примеров
    counts = df.group_by("role_name").agg(pl.len().alias("n"))
    rare = counts.filter(pl.col("n") < d["min_class_count"])["role_name"].to_list()
    df = df.with_columns(
        pl.when(pl.col("role_name").is_in(rare)).then(pl.lit("прочее")).otherwise(pl.col("role_name")).alias("role_name"))

    roles = sorted(df["role_name"].unique().to_list())
    role_map = {r: i for i, r in enumerate(roles)}
    df = df.with_columns(
        pl.col("experience").replace_strict({g: i for i, g in enumerate(GRADES)}, return_dtype=pl.Int64).alias("grade"),
        pl.col("role_name").replace_strict(role_map, return_dtype=pl.Int64).alias("role"),
    )
    print("после чистки: %d вакансий, %d направлений" % (df.height, len(roles)))
    return df, roles


def split(df, cfg, seed):
    """Стратифицированный сплит по паре (грейд, роль): иначе редкие сочетания
    вроде 'сеньор-тестировщик' могут целиком уехать в тест."""
    d = cfg["data"]
    rng = np.random.default_rng(seed)
    df = df.with_columns(pl.Series("u", rng.random(df.height)))
    df = df.with_columns(pl.col("u").rank("ordinal").over(["grade", "role"]).alias("rk"),
                         pl.len().over(["grade", "role"]).alias("n"))
    frac = pl.col("rk") / pl.col("n")
    test = df.filter(frac <= d["test_size"])
    val = df.filter((frac > d["test_size"]) & (frac <= d["test_size"] + d["val_size"]))
    train = df.filter(frac > d["test_size"] + d["val_size"])
    drop = ["u", "rk", "n"]
    print("сплит: train %d, val %d, test %d" % (train.height, val.height, test.height))
    return train.drop(drop), val.drop(drop), test.drop(drop)
