import os

from .keyword_search import InvertedIndex
from .search_utils import INDEX_PICKLE_PATH, load_movies
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PICKLE_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        bm25_scores = normalize([result["score"] for result in bm25_results])
        semantic_scores = normalize([result["score"] for result in semantic_results])

        doc_map = {}
        for result, score in zip(bm25_results, bm25_scores):
            doc_map[result["id"]] = {"doc": result, "bm25": score, "semantic": 0.0}

        for result, score in zip(semantic_results, semantic_scores):
            if result["id"] in doc_map:
                doc_map[result["id"]]["semantic"] = score
            else:
                doc_map[result["id"]] = {"doc": result, "bm25": 0.0, "semantic": score}

        results = []
        for k, v in doc_map.items():
            doc_map[k]["hybrid_score"] = hybrid_score(v["bm25"], v["semantic"], alpha)  # fmt: skip
            results.append(v)

        return sorted(results, key=lambda item: item["hybrid_score"], reverse=True)[:limit]  # fmt: skip

    def rrf_search(self, query: str, k: int, limit: int = 10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(scores: list[float]) -> list[float]:
    min_score, max_score = min(scores), max(scores)
    diff = max_score - min_score
    return [(score - min_score) / diff for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search_command(query: str, alpha: float, limit: int = 5) -> list[dict]:
    hs = HybridSearch(load_movies())
    return hs.weighted_search(query, alpha, limit)
