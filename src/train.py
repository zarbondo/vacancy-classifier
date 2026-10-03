# обучение двухголовой модели: общий лосс = грейд + направление
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class VacancyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].to_list()
        self.grade = df["grade"].to_numpy()
        self.role = df["role"].to_numpy()
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                "grade": torch.tensor(self.grade[i]), "role": torch.tensor(self.role[i])}


def class_weights(y, n, device):
    """Обратные частоты: грейдов 'без опыта' на hh в разы меньше, чем 'от 3 лет'."""
    cnt = np.bincount(y, minlength=n).astype(np.float32)
    w = len(y) / (n * np.maximum(cnt, 1))
    w[cnt == 0] = 0.0
    return torch.tensor(w, device=device)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    gp, rp, gt, rt = [], [], [], []
    for b in loader:
        g, r = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        gp.append(g.argmax(1).cpu().numpy()); rp.append(r.argmax(1).cpu().numpy())
        gt.append(b["grade"].numpy()); rt.append(b["role"].numpy())
    return (np.concatenate(gp), np.concatenate(rp), np.concatenate(gt), np.concatenate(rt))


def train(model, tokenizer, train_df, val_df, cfg, n_grades, n_roles, device):
    t = cfg["train"]
    tr = DataLoader(VacancyDataset(train_df, tokenizer, cfg["model"]["max_len"]),
                    batch_size=t["batch_size"], shuffle=True, drop_last=False)
    va = DataLoader(VacancyDataset(val_df, tokenizer, cfg["model"]["max_len"]), batch_size=t["batch_size"])

    # энкодеру маленький шаг, свежим головам — большой
    head_ids = {id(p) for h in (model.grade_head, model.role_head) for p in h.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": t["lr"]},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": t["head_lr"]},
    ], weight_decay=t["weight_decay"])
    steps = max(1, len(tr) * t["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[t["lr"], t["head_lr"]], total_steps=steps, pct_start=t["warmup_ratio"])

    gw = class_weights(train_df["grade"].to_numpy(), n_grades, device) if t["class_balance"] else None
    rw = class_weights(train_df["role"].to_numpy(), n_roles, device) if t["class_balance"] else None
    loss_g, loss_r = nn.CrossEntropyLoss(weight=gw), nn.CrossEntropyLoss(weight=rw)

    from sklearn.metrics import f1_score
    best, best_state = -1.0, None
    model.to(device)
    for ep in range(t["epochs"]):
        model.train()
        total = 0.0
        for b in tr:
            opt.zero_grad()
            g, r = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = (t["grade_weight"] * loss_g(g, b["grade"].to(device))
                    + t["role_weight"] * loss_r(r, b["role"].to(device)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item() * len(b["grade"])
        gp, rp, gt, rt = predict(model, va, device)
        f_g = f1_score(gt, gp, average="macro"); f_r = f1_score(rt, rp, average="macro")
        mean = (f_g + f_r) / 2
        print("эпоха %d  loss %.4f  val macro-F1: грейд %.4f, направление %.4f" % (ep + 1, total / len(tr.dataset), f_g, f_r))
        if mean > best:  # лучшую эпоху выбираем по val, тест не трогаем
            best = mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model
