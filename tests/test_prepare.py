import polars as pl

from src.data.prepare import GRADES, clean, prepare, split

CFG = {"data": {"min_text_len": 5, "min_class_count": 2, "test_size": 0.2, "val_size": 0.2}}


def make(n_per_group=10):
    rows = []
    for g in GRADES:
        for role in ["Программист", "Аналитик"]:
            for i in range(n_per_group):
                rows.append({"id": len(rows), "text": f"{role} {g} вакансия номер {len(rows)}",
                             "experience": g, "role_id": "1", "role_name": role})
    return pl.DataFrame(rows)


def test_clean_strips_tags():
    assert clean("<b>Python</b>  \n разработчик") == "Python разработчик"


def test_rare_roles_go_to_other():
    df = make()
    df = pl.concat([df, df.head(1).with_columns(pl.lit("Таролог").alias("role_name"),
                                                pl.lit("редкая вакансия таролога").alias("text"))])
    out, roles = prepare(df, CFG)
    assert "Таролог" not in roles and "прочее" in roles


def test_short_texts_dropped():
    df = make()
    df = pl.concat([df, df.head(1).with_columns(pl.lit("ааа").alias("text"))])
    out, _ = prepare(df, CFG)
    assert out.filter(pl.col("text") == "ааа").height == 0


def test_grades_are_ordered():
    out, _ = prepare(make(), CFG)
    m = dict(zip(out["experience"].to_list(), out["grade"].to_list()))
    assert [m[g] for g in GRADES] == [0, 1, 2, 3]


def test_split_is_disjoint_and_stratified():
    out, _ = prepare(make(20), CFG)
    tr, va, te = split(out, CFG, seed=0)
    ids = [set(x["id"].to_list()) for x in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == out.height
    assert set(te["grade"].to_list()) == set(range(len(GRADES)))
