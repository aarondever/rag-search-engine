import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
SCORE_PRECISION = 2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MOVIE_DATA_PATH = os.path.join(DATA_DIR, "movies.json")
STOPWORDS_DATA_PATH = os.path.join(DATA_DIR, "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(DATA_DIR, "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PICKLE_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PICKLE_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PICKLE_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTHS_PICKLE_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")
MOVIE_EMBEDDINGS_NPY_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_NPY_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_NPY_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies() -> list[dict[str, Any]]:
    with open(MOVIE_DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_DATA_PATH, "r") as f:
        return f.read().splitlines()


def load_golden_dataset() -> list[dict[str, Any]]:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)
    return data["test_cases"]
