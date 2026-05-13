import numpy as np
from lib.search_utils import load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(
        self, documents: list[dict], model_name: str = "clip-ViT-B-32"
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts)

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        embeddings = self.model.encode([img])
        return embeddings[0]

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        results = []
        for i, embedding in enumerate(self.text_embeddings):
            cosine_similarity = np.dot(image_embedding, embedding) / (
                np.linalg.norm(image_embedding) * np.linalg.norm(embedding)
            )
            results.append(
                {
                    "id": self.documents[i]["id"],
                    "title": self.documents[i]["title"],
                    "description": self.documents[i]["description"],
                    "similarity": cosine_similarity,
                }
            )

        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]


def verify_image_embeddings(image_path: str) -> None:
    search = MultimodalSearch(load_movies())
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def search_with_image(image_path: str) -> list[dict]:
    search = MultimodalSearch(load_movies())
    return search.search_with_image(image_path)
