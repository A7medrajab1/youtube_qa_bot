"""
SentenceTransformer wrapper compatible with LangChain's FAISS.
"""
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """LangChain-compatible wrapper around SentenceTransformer."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].tolist()