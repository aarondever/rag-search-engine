import os
import re

import numpy as np
from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer

from .search_utils import MOVIE_EMBEDDINGS_NPY_PATH, load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        text = text.strip()
        if len(text) == 0:
            raise ValueError("Text argument cannot be empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        sentences = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            sentences.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(sentences, show_progress_bar=True)
        with open(MOVIE_EMBEDDINGS_NPY_PATH, "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def load_or_create_embedding(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        sentences = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            sentences.append(f"{doc['title']}: {doc['description']}")

        if os.path.exists(MOVIE_EMBEDDINGS_NPY_PATH):
            with open(MOVIE_EMBEDDINGS_NPY_PATH, "rb") as f:
                self.embeddings = np.load(f)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[tuple]:
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embedding = self.generate_embedding(query)
        scores = []
        # Calculate cosine similarity between the query embedding and each document embedding
        for i in range(len(self.embeddings)):
            scores.append(
                (cosine_similarity(embedding, self.embeddings[i]), self.documents[i])
            )

        results = []
        # Sort the list by similarity score in descending order
        for v in sorted(scores, key=lambda item: item[0], reverse=True):
            results.append(v)
            if len(results) >= limit:
                break

        return results


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embedding(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search(query: str, limit: int) -> None:
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embedding(documents)

    results = ss.search(query, limit)
    for i in range(len(results)):
        print(f"{i + 1}. {results[i][1]['title']} (score: {results[i][0]})")
        print(results[i][1]["description"], "\n")


def cosine_similarity(vec1: ArrayLike, vec2: ArrayLike) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunk(text: str, chunk_size: int, overlap: int) -> None:
    words = text.strip().split()
    chunks, current, current_len = [], [], 0

    for words in words:
        if current and current_len + 1 > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] + [words] if overlap else [words]
            current_len = len(current)
        else:
            current.append(words)
            current_len += 1

    if current:
        chunks.append(" ".join(current))

    print(f"Chunking {len(text)} characters")
    for i in range(len(chunks)):
        print(f"{i + 1}. {chunks[i]}")


def semantic_chunk(text: str, chunk_size: int, overlap: int) -> None:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        if current and current_len + 1 > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] + [sentence] if overlap else [sentence]
            current_len = len(current)
        else:
            current.append(sentence)
            current_len += 1

    if current:
        chunks.append(" ".join(current))

    print(f"Semantically Chunking {len(text)} characters")
    for i in range(len(chunks)):
        print(f"{i + 1}. {chunks[i]}")
