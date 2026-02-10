import json
import os

import numpy as np
from lib.search_utils import MOVIE_DATA_PATH, MOVIE_EMBEDDINGS_NPY_PATH
from sentence_transformers import SentenceTransformer


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
    with open(MOVIE_DATA_PATH, "r") as f:
        data = json.load(f)

    documents = data["movies"]
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
