# классификация вакансий: грейд и направление по тексту
#   python main.py collect   — собрать вакансии с hh.ru
#   python main.py train     — обучить и оценить (бейзлайн + rubert), кластеризовать, выгрузить в onnx
#   python main.py --smoke   — быстрый прогон всего на синтетике, без сети
import argparse
import time
from pathlib import Path

import polars as pl
import torch
import yaml

from src import cluster, evaluate, export_onnx, train as train_mod
from src.data import prepare, synthetic
from src.data.prepare import GRADE_RU, GRADES
from src.models import baseline, bert
from src.utils import report


def load_cfg(args):
    path = "configs/smoke.yaml" if args.smoke else args.config
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="train", choices=["collect", "train"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true", help="прогон на синтетике, чтобы проверить пайплайн")
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.command == "collect":
        from src.data.collect import collect
        collect(cfg)
        return

    # данные
    if "synthetic" in cfg["data"]:
        raw = synthetic.generate(cfg["data"]["synthetic"]["n"], seed)
    else:
        path = Path(cfg["data"]["path"])
        if not path.exists():
            raise SystemExit("нет %s — сначала запусти: python main.py collect" % path)
        raw = pl.read_parquet(path)
    df, roles = prepare.prepare(raw, cfg)
    tr, va, te = prepare.split(df, cfg, seed)
    grade_names = [GRADE_RU[g] for g in GRADES]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("устройство: %s" % device)

    # бейзлайн
    print("\n== бейзлайн: TF-IDF + логрег")
    base_grade = baseline.fit_predict(tr["text"].to_list(), tr["grade"].to_numpy(), te["text"].to_list(), seed)
    base_role = baseline.fit_predict(tr["text"].to_list(), tr["role"].to_numpy(), te["text"].to_list(), seed)

    # трансформер
    print("\n== rubert")
    tokenizer, model = bert.build(cfg["model"], len(GRADES), len(roles), corpus=tr["text"].to_list())
    model = train_mod.train(model, tokenizer, tr, va, cfg, len(GRADES), len(roles), device)
    from torch.utils.data import DataLoader
    te_loader = DataLoader(train_mod.VacancyDataset(te, tokenizer, cfg["model"]["max_len"]),
                           batch_size=cfg["train"]["batch_size"])
    gp, rp, gt, rt = train_mod.predict(model, te_loader, device)

    # метрики
    metrics = {
        "grade": {"TF-IDF + логрег": evaluate.scores(te["grade"].to_numpy(), base_grade),
                  "rubert-tiny2": evaluate.scores(gt, gp)},
        "role": {"TF-IDF + логрег": evaluate.scores(te["role"].to_numpy(), base_role),
                 "rubert-tiny2": evaluate.scores(rt, rp)},
        "grade_adjacent_accuracy": {"TF-IDF + логрег": evaluate.adjacent_accuracy(te["grade"].to_numpy(), base_grade),
                                    "rubert-tiny2": evaluate.adjacent_accuracy(gt, gp)},
        "n_roles": len(roles), "n_test": te.height, "roles": roles,
    }
    print("\nГрейд:\n" + evaluate.table(metrics["grade"]))
    print("\nНаправление:\n" + evaluate.table(metrics["role"]))
    print("\nгрейд, попадание в соседний класс: бейзлайн %.4f, rubert %.4f"
          % (metrics["grade_adjacent_accuracy"]["TF-IDF + логрег"], metrics["grade_adjacent_accuracy"]["rubert-tiny2"]))
    (out / "report_grade.txt").write_text(evaluate.report(gt, gp, grade_names), encoding="utf-8")
    (out / "report_role.txt").write_text(evaluate.report(rt, rp, roles), encoding="utf-8")
    (out / "plots").mkdir(exist_ok=True)
    report.confusion(gt, gp, grade_names, out / "plots" / "confusion_grade.png", "Грейд: rubert-tiny2")

    # кластеризация
    if not args.no_cluster:
        print("\n== кластеризация эмбеддингов")
        sample = df.sample(min(cfg["cluster"]["sample"], df.height), seed=seed)
        emb = cluster.embed(model, tokenizer, sample["text"].to_list(), cfg["model"]["max_len"], device)
        sample, clusters, cl_stats = cluster.run(sample, emb, cfg, seed)
        report.save_clusters(clusters, out)
        # кластеры, которые не сводятся к одной роли hh — самое интересное
        mixed = clusters.filter(pl.col("role_share") < 0.6)
        metrics["clusters"] = {**cl_stats, "mixed": mixed.height}
        print("из них не сводятся к одной роли hh: %d" % mixed.height)

    # onnx
    if not args.no_onnx:
        print("\n== onnx")
        path = export_onnx.export(model, tokenizer, cfg, cfg["export"]["onnx_path"])
        metrics["latency"] = export_onnx.benchmark(path, tokenizer, cfg, cfg["export"]["bench_runs"])

    report.save(metrics, out)
    if not args.smoke:
        block = ("**Грейд**\n\n" + evaluate.table(metrics["grade"])
                 + "\n\n**Направление** (%d классов)\n\n" % len(roles) + evaluate.table(metrics["role"]))
        if "latency" in metrics:
            block += "\n\nЗадержка ONNX на CPU: %.1f мс в среднем, p95 %.1f мс." % (
                metrics["latency"]["mean_ms"], metrics["latency"]["p95_ms"])
        if "clusters" in metrics:
            block += " Кластеров найдено: %d, из них %d не сводятся к одной роли hh.ru." % (
                metrics["clusters"]["n_clusters"], metrics["clusters"]["mixed"])
        report.update_readme(block)

    torch.save(model.state_dict(), out / "model.pt")
    print("\nготово за %.1f мин, результаты в %s/" % ((time.time() - t0) / 60, out))


if __name__ == "__main__":
    main()
