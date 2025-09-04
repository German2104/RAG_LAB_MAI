#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Общие константы для всех модулей RAG-системы
"""

import os
from pathlib import Path

# === Пути ===
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)

DB_PATH = str(DB_DIR / "milvus.db")

# === Milvus ===
COLLECTION = "pdf_embeddings"
VECTOR_FIELD = "vector"

# === Embeddings ===
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
DIMENSION = 1024

# === Deploy service ===
SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8000
SERVICE_URL = f"http://localhost:{SERVICE_PORT}"

# === Бот ===
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# === Метрики поиска ===
SEARCH_METRIC = "IP"  # Inner Product (т.к. векторы нормализованы)
TOP_K_DEFAULT = 5