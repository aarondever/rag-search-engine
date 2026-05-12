import json
import os
from time import sleep

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

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
    query: str,
    k: int = 60,
    limit: int = 10,
    enhance: str | None = None,
    rerank: str | None = None,
    evaluate: bool = False,
) -> list[dict]:
    hs = HybridSearch(load_movies())

    print(f"[search] Original query: '{query}'")

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

    results = hs.rrf_search(query, k, 5 * limit if rerank is not None else limit)
    print(
        f"[search] RRF results ({len(results)}): {[r['doc']['title'] for r in results]}\n"
    )

    if evaluate:
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

            Query: "{query}"

            Results:
            {
                chr(10).join(
                    f"{i + 1}. {v['doc'].get('title', '')} - {v['doc'].get('description', '')[:150]}"
                    for i, v in enumerate(results)
                )
            }

            Scale:
            - 3: Highly relevant
            - 2: Relevant
            - 1: Marginally relevant
            - 0: Not relevant

            Do NOT give any numbers other than 0, 1, 2, or 3.

            Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

            [2, 0, 3, 2, 0, 1]""",
        )
        scores = json.loads(response.text)
        for i, score in enumerate(scores):
            print(f"{i + 1}. {results[i]['doc']['title']}: {score}/3")

    match rerank:
        case "individual":
            for result in results:
                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {result["doc"].get("title", "")} - {result["doc"].get("description", "")[:150]}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Output ONLY the number in your response, no other text or explanation.

                    Score:""",
                )
                sleep(3)
                result["rerank_score"] = float(response.text)

            results = sorted(results, key=lambda item: item["rerank_score"], reverse=True)[:limit]  # fmt: skip

        case "batch":
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=f"""Rank the movies listed below by relevance to the following search query.

                    Query: "{query}"

                    Movies:
                    {
                    chr(10).join(
                        f"{i + 1}. [ID: {v['doc']['id']}] {v['doc'].get('title', '')} - {v['doc'].get('description', '')[:150]}"
                        for i, v in enumerate(results)
                    )
                }

                    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

                    For example:
                    [75, 12, 34, 2, 1]

                    Ranking:""",
            )

            rankings = json.loads(response.text)
            rank_position = {movie_id: i for i, movie_id in enumerate(rankings)}
            for result in results:
                result["batch_rerank_score"] = rank_position.get(
                    result["doc"]["id"], len(rankings)
                )

            results = sorted(results, key=lambda item: item["batch_rerank_score"])[:limit]  # fmt: skip

        case "cross_encoder":
            pairs = [
                (query, f"{doc.get('title', '')} - {doc.get('description', '')}")
                for doc in [v["doc"] for v in results]
            ]
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            # `predict` returns a list of numbers, one for each pair
            scores = cross_encoder.predict(pairs)

            for i, rank in enumerate(scores):
                results[i]["cross_encoder_score"] = rank

            results = sorted(results, key=lambda item: item["cross_encoder_score"], reverse=True)[:limit]  # fmt: skip

    print(
        f"[search] Final results ({len(results)}): {[r['doc']['title'] for r in results]}\n"
    )
    return results
