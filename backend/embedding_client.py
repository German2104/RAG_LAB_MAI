# backend/embedding_client.py
"""
Клиент для отправки запросов к деплою JinaAI эмбеддингов
"""
import requests
import numpy as np
from typing import List, Union

# URL деплоя эмбеддингов
EMBEDDING_SERVICE_URL = "http://localhost:8100/v1/models/jinaai-embeddings:predict"

def get_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Отправляет запрос к деплою JinaAI и возвращает эмбеддинги.
    
    Args:
        texts: строка или список строк для получения эмбеддингов
        
    Returns:
        np.ndarray: матрица эмбеддингов формы (N, 1024) для списка или (1024,) для одной строки
    """
    # Приводим к списку для единообразия
    if isinstance(texts, str):
        input_texts = [texts]
        single_text = True
    else:
        input_texts = texts
        single_text = False
    
    # Формируем запрос
    payload = {"inputs": input_texts}
    
    try:
        response = requests.post(EMBEDDING_SERVICE_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        embeddings = np.array(result["embeddings"], dtype=np.float32)
        
        # Если был один текст, возвращаем одномерный массив
        if single_text:
            return embeddings[0]
        
        return embeddings
        
    except requests.RequestException as e:
        raise RuntimeError(f"Ошибка при запросе к эмбеддинг-сервису: {e}")
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Некорректный ответ от эмбеддинг-сервиса: {e}")

def emb_text(text: str) -> np.ndarray:
    """
    Совместимая функция для получения эмбеддинга одного текста.
    Возвращает L2-нормированный вектор размерности 1024.
    """
    return get_embeddings(text)

def emb_texts(texts: List[str]) -> np.ndarray:
    """
    Совместимая функция для получения эмбеддингов списка текстов.
    Возвращает матрицу (N, 1024) L2-нормированных эмбеддингов.
    """
    return get_embeddings(texts)
