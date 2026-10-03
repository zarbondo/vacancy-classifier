# rubert-tiny2 с двумя головами: грейд и направление.
# энкодер общий — задачи родственные, и так модель одна вместо двух
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class TwoHeadClassifier(nn.Module):
    def __init__(self, encoder, hidden, n_grades, n_roles, dropout):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.grade_head = nn.Linear(hidden, n_grades)
        self.role_head = nn.Linear(hidden, n_roles)

    def embed(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # mean pooling по неподбитым токенам: у tiny-моделей работает заметно лучше, чем [CLS]
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        h = self.drop(self.embed(input_ids, attention_mask))
        return self.grade_head(h), self.role_head(h)


def build_tiny(cfg, corpus):
    """Мини-BERT со случайными весами и свой word-piece токенизатор — только для --smoke."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    d = Path(tempfile.mkdtemp())
    (d / "corpus.txt").write_text("\n".join(corpus), encoding="utf-8")
    tok = BertWordPieceTokenizer(lowercase=True)
    tok.train([str(d / "corpus.txt")], vocab_size=cfg["tiny"]["vocab_size"], min_frequency=1)
    (d / "tok").mkdir()
    tok.save_model(str(d / "tok"))
    tokenizer = BertTokenizerFast.from_pretrained(str(d / "tok"))
    t = dict(cfg["tiny"])
    t["vocab_size"] = tokenizer.vocab_size
    return tokenizer, BertModel(BertConfig(**t))


def build(cfg, n_grades, n_roles, corpus=None):
    if cfg["name"] == "__tiny__":
        tokenizer, encoder = build_tiny(cfg, corpus)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        encoder = AutoModel.from_pretrained(cfg["name"])
    hidden = encoder.config.hidden_size
    model = TwoHeadClassifier(encoder, hidden, n_grades, n_roles, cfg["dropout"])
    print("модель: %s, параметров %.1fM" % (cfg["name"], sum(p.numel() for p in model.parameters()) / 1e6))
    return tokenizer, model
