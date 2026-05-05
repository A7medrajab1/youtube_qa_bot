"""
Main pipeline class that orchestrates everything.
Supports switching between providers and models on the fly.
"""
from langchain_core.output_parsers import StrOutputParser

from src.transcript import fetch_transcript, format_transcript
from src.embeddings import LocalEmbeddings
from src.vector_store import chunk_text, build_faiss_index
from src.llm_setup import get_llm
from src.prompts import get_summary_prompt, get_qa_prompt
from config import EMBEDDING_MODEL_NAME, TOP_K, DEFAULT_PROVIDER, DEFAULT_MODEL


class YouTubeBot:
    def __init__(self):
        # Embeddings loaded once (heavy)
        self.embeddings = LocalEmbeddings(EMBEDDING_MODEL_NAME)

        # LLM state — initialized lazily
        self.current_provider: str | None = None
        self.current_model: str | None = None
        self.llm = None
        self.summary_chain = None
        self.qa_chain = None

        # Per-video state
        self.transcript_text: str | None = None
        self.transcript_language: str | None = None
        self.faiss_index = None
        self.current_url: str | None = None

        # Initialize with defaults
        self.set_llm(DEFAULT_PROVIDER, DEFAULT_MODEL)

    def set_llm(self, provider: str, model: str) -> str:
        """Switch the active LLM (provider + model)."""
        if provider == self.current_provider and model == self.current_model:
            return f"⚙️ Already using {provider}/{model}"

        self.llm = get_llm(provider, model)
        self.summary_chain = get_summary_prompt() | self.llm | StrOutputParser()
        self.qa_chain = get_qa_prompt() | self.llm | StrOutputParser()
        self.current_provider = provider
        self.current_model = model
        return f"⚙️ Switched to {provider} / {model}"

    def load_video(self, url: str) -> str:
        if not url:
            raise ValueError("Please provide a YouTube URL")

        if url == self.current_url and self.faiss_index is not None:
            return f"✅ Video already loaded (language: {self.transcript_language})"

        raw, lang_code = fetch_transcript(url)
        if not raw:
            raise ValueError("No transcript available for this video")

        self.transcript_text = format_transcript(raw)
        self.transcript_language = lang_code
        chunks = chunk_text(self.transcript_text)
        self.faiss_index = build_faiss_index(chunks, self.embeddings)
        self.current_url = url

        return f"✅ Video loaded (language: {lang_code}, {len(chunks)} chunks indexed)"

    def summarize(self) -> str:
        if not self.transcript_text:
            raise ValueError("Load a video first")
        return self.summary_chain.invoke({"transcript": self.transcript_text})

    def ask(self, question: str, k: int = TOP_K) -> str:
        if not self.faiss_index:
            raise ValueError("Load a video first")
        if not question or not question.strip():
            raise ValueError("Please enter a question")

        docs = self.faiss_index.similarity_search(question, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        return self.qa_chain.invoke({"context": context, "question": question})