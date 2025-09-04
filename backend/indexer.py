from pathlib import Path
from typing import List, Dict
import numpy as np
from pymilvus import MilvusClient, DataType
import pymupdf4llm
from docx import Document
from .embedding_client import emb_text

# ====== Конфиг БД/модели ======
DB_PATH      = "./milvus.db"
COLLECTION   = "pdf_embeddings"
VECTOR_FIELD = "vector"
DIMENSION    = 1024  # для jinaai/jina-embeddings-v3
TEXT_FIELD   = "text"
ID_FIELD     = "id"


# ====== Парсеры ======
def extract_pdf(pdf_path: str) -> List[Dict]:
    """
    Возвращает список [{text: str, metadata: {page: int}}]
    Чанкуем помарково встроенным механизмом pymupdf4llm.
    """
    return pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

def extract_docx(docx_path: str) -> List[Dict]:
    """
    Возвращает список [{text: str, metadata: {page: int}}] — page == номер абзаца.
    """
    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return [{"text": para, "metadata": {"page": i}} for i, para in enumerate(paragraphs)]

def extract_txt(txt_path: str) -> List[Dict]:
    """
    Разбиваем TXT по абзацам (пустые строки — разделители).
    """
    text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return [{"text": b, "metadata": {"page": i}} for i, b in enumerate(blocks)]

# ====== Подготовка коллекции ======
def _ensure_collection(client: MilvusClient):
    if not client.has_collection(COLLECTION):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name=ID_FIELD, datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name=VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        # text — VARCHAR; Milvus поддерживает до 65,535 (тут укажем максимум)
        schema.add_field(field_name=TEXT_FIELD, datatype=DataType.VARCHAR, max_length=65535)

        client.create_collection(
            collection_name=COLLECTION,
            schema=schema,
            consistency_level="Strong",
        )

        # Индекс по вектору (AUTOINDEX + Inner Product)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="IP",
            params={"nlist": 1024},
        )
        client.create_index(collection_name=COLLECTION, index_params=index_params)

# ====== Индексация файла ======
def index_file(file_path: str) -> None:
    """
    Детектирует тип файла, извлекает текстовые чанки, считает эмбеддинги и вставляет в Milvus.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        chunks = extract_pdf(str(path))
    elif suffix == ".docx":
        chunks = extract_docx(str(path))
    elif suffix == ".txt":
        chunks = extract_txt(str(path))
    else:
        raise ValueError(f"Неподдерживаемый тип файла: {suffix} (поддерживаются .pdf, .docx, .txt)")

    # Инициализируем Milvus Lite
    client = MilvusClient(uri=DB_PATH)
    _ensure_collection(client)

    # Сдвигаем id, чтобы не затирать прошлые вставки
    # Получим текущий размер коллекции
    stats = client.get_collection_stats(collection_name=COLLECTION)
    row_count = int(stats["row_count"]) if "row_count" in stats else 0
    next_id = row_count

    to_insert = []
    for i, item in enumerate(chunks):
        text = item["text"]
        if not text.strip():
            continue
        vec = emb_text(text)
        to_insert.append({
            ID_FIELD: next_id + i,
            VECTOR_FIELD: vec,
            TEXT_FIELD: text,
        })

    if not to_insert:
        return

    client.insert(collection_name=COLLECTION, data=to_insert)