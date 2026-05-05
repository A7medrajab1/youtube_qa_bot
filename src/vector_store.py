"""
Text chunking and FAISS vector store management.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_faiss_index(chunks: list[str], embeddings: Embeddings) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build index from empty chunks")
    return FAISS.from_texts(chunks, embeddings)