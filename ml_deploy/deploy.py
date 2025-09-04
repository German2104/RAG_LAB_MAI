
# ==== JinaAI Embedding Model Server (kserve) ====
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"

from vllm import LLM
import numpy as np

class JinaAIEmbedder:
    def __init__(self):
        self.llm = LLM(
            model=EMBEDDING_MODEL_NAME,
            runner="pooling",
            trust_remote_code=True,
            dtype="auto",
        )
        self.dimension = 1024

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)

    def predict(self, payload: dict) -> dict:
        # payload: {"inputs": ["text1", "text2", ...]}
        texts = payload.get("inputs", [])
        outs = self.llm.embed(texts)
        mat = [o.outputs.embedding for o in outs]
        mat = np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat_norm = (mat / norms).astype(np.float32)
        return {"embeddings": mat_norm.tolist()}

# ==== kserve ModelServer ====
import kserve

class KServeJinaAIModel(kserve.Model):
    def __init__(self, name="jinaai-embeddings"):
        super().__init__(name)
        self.embedder = JinaAIEmbedder()
        self.name = name
        self.ready = True

    def predict(self, payload, **kwargs):
        return self.embedder.predict(payload)

if __name__ == "__main__":
    model = KServeJinaAIModel()
    kserve.ModelServer(http_port=8100, enable_docs_url=True).start(models=[model])