# backend/config.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# --- резолвинг корня проекта ---
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- .env (опционально) ---
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass

# === Пути ===
BASE_DIR = ROOT_DIR
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)

DB_PATH = str(DB_DIR / "milvus.db")  # единый путь

# === Milvus ===
COLLECTION = os.getenv("COLLECTION_NAME", "pdf_embeddings")
VECTOR_FIELD = os.getenv("VECTOR_FIELD", "vector")

# === Embeddings ===
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jinaai/jina-embeddings-v3")
DIMENSION = int(os.getenv("DIMENSION", "1024"))

# === Deploy service ===
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8000"))
SERVICE_URL  = os.getenv("SERVICE_URL", f"http://localhost:{SERVICE_PORT}")

# === Бот ===
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# === Метрики поиска ===
SEARCH_METRIC = os.getenv("SEARCH_METRIC", "IP")
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

# === GigaChat (если используешь OpenAI-совместимое API) ===
GIGACHAT_API_URL = os.getenv("GIGACHAT_API_URL")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

# === Telegram ===
BOT_TOKEN = os.getenv("BOT_TOKEN")