import os

from dotenv import load_dotenv
from google import genai

from .keyword_search import InvertedIndex
from .search_utils import INDEX_PICKLE_PATH, load_movies
from .semantic_search import ChunkedSemanticSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


client = genai.Client(api_key=api_key)


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

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        doc_map = {}
        for i, v in enumerate(bm25_results):
            doc_map[v["id"]] = {"doc": v, "bm25": i + 1, "semantic": 0}

        for i, v in enumerate(semantic_results):
            if v["id"] in doc_map:
                doc_map[v["id"]]["semantic"] = i
            else:
                doc_map[v["id"]] = {"doc": v, "bm25": 0, "semantic": i + 1}

        results = []
        for k, v in doc_map.items():
            doc_map[k]["rrf_score"] = rrf_score(v["bm25"] + v["semantic"], k)  # fmt: skip
            results.append(v)

        return sorted(results, key=lambda item: item["rrf_score"], reverse=True)[:limit]  # fmt: skip


def normalize(scores: list[float]) -> list[float]:
    min_score, max_score = min(scores), max(scores)
    diff = max_score - min_score
    return [(score - min_score) / diff for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def weighted_search_command(query: str, alpha: float, limit: int = 5) -> list[dict]:
    hs = HybridSearch(load_movies())
    return hs.weighted_search(query, alpha, limit)


def rrf_search_command(
    query: str, k: int = 60, limit: int = 10, enhance: str | None = None
) -> list[dict]:
    match enhance:
        case "spell":
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """,
            )
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text

        case "rewrite":
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """,
            )
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text

        case "expand":
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=f"""Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """,
            )
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text

    hs = HybridSearch(load_movies())
    return hs.rrf_search(query, k, limit)
