# macro-F1 по обеим головам + отчёт по классам.
# macro, а не accuracy: классы несбалансированы, и accuracy переоценит модель,
# которая просто угадывает самый частый грейд
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def scores(y_true, y_pred):
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "accuracy": float(accuracy_score(y_true, y_pred))}


def adjacent_accuracy(y_true, y_pred):
    """Грейды упорядочены, поэтому ошибка на соседний грейд (мидл вместо сеньора)
    гораздо мягче, чем 'без опыта' вместо лида. Считаем и это."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def report(y_true, y_pred, names):
    return classification_report(y_true, y_pred, target_names=names, digits=3, zero_division=0)


def table(rows, ks=("macro_f1", "weighted_f1", "accuracy")):
    head = "| модель | " + " | ".join(k.replace("_", "-") for k in ks) + " |"
    out = [head, "|---" * (len(ks) + 1) + "|"]
    for name, m in rows.items():
        out.append(f"| {name} | " + " | ".join(f"{m[k]:.4f}" for k in ks) + " |")
    return "\n".join(out)
