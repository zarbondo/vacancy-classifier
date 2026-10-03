# экспорт в ONNX и замер задержки на CPU: в проде GPU под такую модель держать незачем
import time
from pathlib import Path

import numpy as np
import torch


def export(model, tokenizer, cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    enc = tokenizer("тестовая вакансия", truncation=True, max_length=cfg["model"]["max_len"],
                    padding="max_length", return_tensors="pt")
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["grade_logits", "role_logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                      "grade_logits": {0: "batch"}, "role_logits": {0: "batch"}},
        opset_version=18,
    )
    print("onnx сохранён: %s (%.1f МБ)" % (path, path.stat().st_size / 1e6))
    return path


def benchmark(path, tokenizer, cfg, runs):
    """Задержка на одном запросе — то, что увидит пользователь бота."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    enc = tokenizer("ищем python-разработчика с опытом от трёх лет", truncation=True,
                    max_length=cfg["model"]["max_len"], padding="max_length", return_tensors="np")
    feed = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
    for _ in range(5):
        sess.run(None, feed)  # прогрев
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    res = {"mean_ms": float(times.mean()), "p50_ms": float(np.percentile(times, 50)),
           "p95_ms": float(np.percentile(times, 95))}
    print("задержка на CPU: среднее %.1f мс, p50 %.1f, p95 %.1f" % (res["mean_ms"], res["p50_ms"], res["p95_ms"]))
    return res
