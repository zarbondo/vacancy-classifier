# кластеризация эмбеддингов: ищем группы вакансий, которых нет в фильтрах hh.ru.
# UMAP сначала — HDBSCAN плохо работает в сотнях измерений (проклятие размерности)
import numpy as np
import polars as pl
import torch


@torch.no_grad()
def embed(model, tokenizer, texts, max_len, device, batch=64):
    model.eval().to(device)
    out = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        out.append(model.embed(enc["input_ids"], enc["attention_mask"]).cpu().numpy())
    return np.vstack(out)


def top_words(texts, k=6):
    """Чем кластер отличается от остальных — по TF-IDF внутри кластера."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(max_features=5000, min_df=2)
    try:
        m = v.fit_transform(texts)
    except ValueError:
        return []
    weights = np.asarray(m.mean(axis=0)).ravel()
    names = np.array(v.get_feature_names_out())
    return names[np.argsort(weights)[::-1][:k]].tolist()


def run(df, emb, cfg, seed):
    import hdbscan
    import umap

    c = cfg["cluster"]
    reducer = umap.UMAP(random_state=seed, metric="cosine", **c["umap"])
    low = reducer.fit_transform(emb)
    labels = hdbscan.HDBSCAN(**c["hdbscan"]).fit_predict(low)

    df = df.with_columns(pl.Series("cluster", labels))
    noise = float((labels == -1).mean())
    n_clusters = len({l for l in labels if l != -1})
    print("кластеров: %d, шума: %.1f%%" % (n_clusters, 100 * noise))

    rows = []
    for cl in sorted({l for l in labels if l != -1}):
        part = df.filter(pl.col("cluster") == cl)
        roles = part.group_by("role_name").agg(pl.len().alias("n")).sort("n", descending=True)
        rows.append({
            "cluster": int(cl),
            "size": part.height,
            "top_role": roles["role_name"][0],
            "role_share": round(roles["n"][0] / part.height, 3),  # <0.6 — кластер не сводится к одной роли hh
            "words": ", ".join(top_words(part["text"].to_list())),
        })
    return df, pl.DataFrame(rows).sort("size", descending=True), {"n_clusters": n_clusters, "noise": noise}
