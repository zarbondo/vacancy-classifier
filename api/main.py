# сервис над ONNX-моделью: POST /predict с текстом вакансии → грейд и направление
# запуск: uvicorn api.main:app --host 0.0.0.0 --port 8000
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

MODEL = os.getenv("MODEL_PATH", "results/model.onnx")
TOKENIZER = os.getenv("TOKENIZER", "cointegrated/rubert-tiny2")
META = os.getenv("META_PATH", "results/metrics.json")
MAX_LEN = int(os.getenv("MAX_LEN", "256"))
GRADES = ["без опыта", "1–3 года", "3–6 лет", "более 6 лет"]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # модель грузим один раз на старте, а не на каждый запрос
    if not Path(MODEL).exists():
        raise RuntimeError("нет %s — сначала обучи модель: python main.py train" % MODEL)
    state["sess"] = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    state["tok"] = AutoTokenizer.from_pretrained(TOKENIZER)
    state["roles"] = json.loads(Path(META).read_text(encoding="utf-8"))["roles"] if Path(META).exists() else []
    yield
    state.clear()


app = FastAPI(title="Vacancy classifier", version="1.0", lifespan=lifespan)


class Item(BaseModel):
    text: str = Field(..., min_length=10, description="текст вакансии")


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


@app.get("/health")
def health():
    return {"status": "ok", "roles": len(state.get("roles", []))}


@app.post("/predict")
def predict(item: Item):
    if "sess" not in state:
        raise HTTPException(503, "модель не загружена")
    enc = state["tok"](item.text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="np")
    g, r = state["sess"].run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                    "attention_mask": enc["attention_mask"].astype(np.int64)})
    gp, rp = softmax(g[0]), softmax(r[0])
    roles = state["roles"] or [str(i) for i in range(len(rp))]
    return {
        "grade": {"label": GRADES[int(gp.argmax())], "confidence": round(float(gp.max()), 3)},
        "role": {"label": roles[int(rp.argmax())], "confidence": round(float(rp.max()), 3)},
        "top_roles": [{"label": roles[i], "p": round(float(rp[i]), 3)} for i in np.argsort(rp)[::-1][:3]],
    }
