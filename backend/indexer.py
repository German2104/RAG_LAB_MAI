#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import requests
import numpy as np
import fitz
import pymupdf4llm
import docx
from tqdm import tqdm
from pymilvus import MilvusClient, DataType

from config import DB_PATH, COLLECTION, VECTOR_FIELD, DIMENSION, SERVICE_URL

# -------------------- utils --------------------
def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    md = pymupdf4llm.to_markdown(doc)
    doc.close()
    return clean_ws(md)

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_ws(f.read())

def load_docx(path: str) -> str:
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs if p.text]
    return clean_ws("\n".join(paras))

def extract_text(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": return load_pdf(path), "pdf"
    if ext == ".txt": return load_txt(path), "txt"
    if ext == ".docx": return load_docx(path), "docx"
    raise ValueError(f"Неизвестный формат {ext}")

def chunk_text(text: str, chunk_size=700, overlap=120) -> list[str]:
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        part = words[i:i + chunk_size]
        if not part: break
        chunks.append(" ".join(part))
        if i + chunk_size >= len(words): break
    return chunks

# -------------------- сервис эмбеддингов --------------------
def embed_via_service(texts: list[str]) -> np.ndarray:
    """Отправляем запрос к deploy.py (/embed) и получаем эмбеддинги [N, DIMENSION]."""
    if not texts:
        return np.zeros((0, DIMENSION), dtype=np.float32)
    r = requests.post(f"{SERVICE_URL}/embed", json={"texts": texts}, timeout=120)
    r.raise_for_status()
    arr = np.array(r.json()["embeddings"], dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Ожидался двумерный массив [N,{DIMENSION}], получили ndim={arr.ndim}")
    if arr.shape[1] != DIMENSION:
        raise RuntimeError(f"Ожидался размер [{DIMENSION}] по последней оси, получили {arr.shape}")
    return arr

# -------------------- Milvus --------------------
def ensure_collection(milvus: MilvusClient):
    if milvus.has_collection(COLLECTION):
        return
    schema = MilvusClient.create_schema(
        auto_id=True,                 # <-- включаем авто-ID, чтобы не было коллизий
        enable_dynamic_field=False,
    )
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=512)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=16)
    schema.add_field("chunk_id", DataType.INT64)

    milvus.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        consistency_level="Strong",
        num_shards=2,
    )

    # Индекс под косинус/IP (мы нормализуем на стороне сервиса → IP уместен)
    milvus.create_index(
        collection_name=COLLECTION,
        index_params={
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
            "params": {}
        }
    )

def load_collection(milvus: MilvusClient):
    try:
        milvus.load_collection(collection_name=COLLECTION)
    except Exception:
        # не у всех версий клиента это требуется; игнорируем, если не поддерживается
        pass

# -------------------- главная функция --------------------
def index_file(path: str, chunk_size_words: int = 700, chunk_overlap_words: int = 120, batch_size: int = 16):
    """
    Загружает файл, бьёт на чанки, получает эмбеддинги от deploy.py и индексирует в Milvus.
    Добавляет doc_name/doc_type/chunk_id, auto_id генерируется Milvus.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    text, doc_type = extract_text(path)
    chunks = chunk_text(text, chunk_size=chunk_size_words, overlap=chunk_overlap_words)

    if not chunks:
        print(f"⚠️ Нет текста для индексации в {path}")
        return

    # Получаем эмбеддинги пачками
    embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
        batch = chunks[i:i + batch_size]
        vecs = embed_via_service(batch)   # [B, D]
        embeddings.append(vecs)
    if embeddings:
        embeddings = np.vstack(embeddings)  # [N, D]
    else:
        embeddings = np.zeros((0, DIMENSION), dtype=np.float32)

    # Создаём Milvus Lite / коллекцию
    milvus = MilvusClient(uri=DB_PATH)
    ensure_collection(milvus)

    fname = os.path.basename(path)
    rows = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        rows.append({
            # id не указываем: auto_id=True
            "text": (chunk[:4096] if len(chunk) > 4096 else chunk),
            "doc_name": fname,
            "doc_type": doc_type,
            "chunk_id": int(i),
            VECTOR_FIELD: vec.tolist(),
        })

    # Вставка
    if rows:
        milvus.insert(collection_name=COLLECTION, data=rows)

    # Коллекция готова к поиску
    load_collection(milvus)

    print(f"✅ Indexed {len(chunks)} chunks from {path} into collection '{COLLECTION}'.")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python indexer.py <file.pdf|file.txt|file.docx>")
        sys.exit(1)
    index_file(sys.argv[1])