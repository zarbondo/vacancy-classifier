FROM python:3.11-slim

WORKDIR /app

# в контейнере нужен только инференс: onnxruntime, токенизатор и fastapi.
# torch весит под гигабайт и для рантайма не нужен
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ api/
COPY results/model.onnx results/metrics.json results/

ENV MODEL_PATH=results/model.onnx \
    META_PATH=results/metrics.json \
    TOKENIZER=cointegrated/rubert-tiny2 \
    MAX_LEN=256

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
