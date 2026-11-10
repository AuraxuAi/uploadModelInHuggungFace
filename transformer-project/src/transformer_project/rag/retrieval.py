"""
Retrieval component for RAG. Uses TF-IDF + cosine similarity rather than a
neural embedding model — deliberately dependency-light (scikit-learn only,
no HuggingFace model download required), and genuinely adequate for
keyword-driven retrieval over a small-to-medium document set. Swap in a
real sentence-transformer embedding model later if retrieval quality on
paraphrased/semantic queries (not just keyword overlap) turns out to
matter for your use case — this is a deliberate starting point, not a
permanent ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    source: str


class DocumentStore:
    """Chunk documents, index them with TF-IDF, and retrieve the most
    relevant chunks for a query."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add_document(self, text: str, source: str = "unknown") -> None:
        words = text.split()
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.chunk_size]
            if not chunk_words:
                continue
            self.chunks.append(Chunk(text=" ".join(chunk_words), source=source))

    def build_index(self) -> None:
        if not self.chunks:
            raise ValueError("No documents added — call add_document() first.")
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        if self.vectorizer is None:
            raise RuntimeError("Call build_index() before retrieve().")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]
