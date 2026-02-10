import math
import os
import pickle
import string
from collections import Counter, defaultdict
from typing import Any

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DOC_LENGTHS_PICKLE_PATH,
    DOCMAP_PICKLE_PATH,
    INDEX_PICKLE_PATH,
    TERM_FREQUENCIES_PICKLE_PATH,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        # Tokenize the input
        tokens = tokenize_text(text)

        # Add each token to the index with the document ID
        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    @staticmethod
    def __tokenize_term(term: str) -> str:
        # Tokenize the input
        tokens = tokenize_text(term)
        if not tokens:
            raise Exception("Must have at least one token")
        if len(tokens) > 1:
            raise Exception("Cannot have more than one token")

        return tokens[0]

    def __get_avg_doc_length(self) -> float:
        total_doc_count = len(self.doc_lengths)
        if total_doc_count == 0:
            return 0.0

        return sum(self.doc_lengths.values()) / total_doc_count

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(self.__tokenize_term(term))
        if doc_ids is None:
            return []

        return sorted(list(doc_ids))

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        # Tokenization query
        query_tokens = tokenize_text(query)
        seen, results = set(), []

        for token in query_tokens:
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                if doc_id in seen:
                    continue

                seen.add(doc_id)
                doc = self.docmap[doc_id]
                results.append(doc)
                if len(results) >= limit:
                    return results

        return results

    def get_tf(self, doc_id: int, term: str) -> int:
        tf = self.term_frequencies[doc_id].get(self.__tokenize_term(term))
        if tf is None:
            return 0

        return tf

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

        return idf

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        # Length normalization factor
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_tf

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        idf = math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

        return idf

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        # Tokenization query
        query_tokens = tokenize_text(query)
        scores, results = {}, []

        for token in query_tokens:
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                scores.setdefault(doc_id, 0.0)
                scores[doc_id] += self.bm25(doc_id, token)

        for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            doc = self.docmap[k]
            doc["score"] = v
            results.append(doc)
            if len(results) >= limit:
                break

        return results

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def load(self) -> None:
        with open(INDEX_PICKLE_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_PICKLE_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        with open(TERM_FREQUENCIES_PICKLE_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(DOC_LENGTHS_PICKLE_PATH, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(INDEX_PICKLE_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_PICKLE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TERM_FREQUENCIES_PICKLE_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(DOC_LENGTHS_PICKLE_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)


stopwords = load_stopwords()
punctuation_map = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()


def tokenize_text(text: str) -> list[str]:
    return [
        # Stemming
        stemmer.stem(token)
        # Remove punctuation
        for token in text.lower().translate(punctuation_map).split()
        # Remove stopwords
        if token and token not in stopwords
    ]


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.search(query, limit)


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)


def mb25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def mb25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def mb25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
