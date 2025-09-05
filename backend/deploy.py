#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import uvicorn

from backend.config import EMBEDDING_MODEL_NAME, DIMENSION, SERVICE_HOST, SERVICE_PORT, HF_TOKEN

device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cpu")
)

# === Загружаем модель один раз при старте ===
print(f"🚀 Загружаем модель {EMBEDDING_MODEL_NAME} на {device}...")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
if device == "cuda":
    model = model.half()
model = model.eval().to(device)
print("✅ Модель готова!")

# === FastAPI ===
app = FastAPI(title="Jina Embedding Service")

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]

@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": device, "model": EMBEDDING_MODEL_NAME}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.texts:
        return {"embeddings": []}

    with torch.no_grad():
        toks = tokenizer(
            req.texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        outputs = model(**toks)
        last_hidden = outputs.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1)

        # mean pooling
        masked = last_hidden * mask
        sum_vec = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_vec = sum_vec / lengths

        # L2 нормализация
        mean_vec = torch.nn.functional.normalize(mean_vec, p=2, dim=1)

        embeddings = mean_vec.cpu().to(torch.float32).numpy().tolist()
        return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, reload=False)