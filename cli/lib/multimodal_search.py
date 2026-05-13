import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, path: str) -> np.ndarray:
        img = Image.open(path)
        embeddings = self.model.encode([img])
        return embeddings[0]


def verify_image_embeddings(path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
