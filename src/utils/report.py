# сохранение результатов, графики, обновление таблицы в README
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def save(summary, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def save_clusters(df, out):
    df.write_csv(Path(out) / "clusters.csv")


def confusion(y_true, y_pred, names, path, title):
    n = len(names)
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * n + 3, 1.0 * n + 2.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title(title)
    ax.set_xlabel("предсказано"); ax.set_ylabel("на самом деле")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def update_readme(block, path="README.md"):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    text = re.sub(r"<!-- results -->.*?<!-- /results -->",
                  lambda _: f"<!-- results -->\n{block}\n<!-- /results -->", text, flags=re.S)
    p.write_text(text, encoding="utf-8")
