#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import string
from collections import Counter
from typing import Any

from nltk.stem import PorterStemmer

CACHE_DIR_PATH = "cache"
INDEX_PICKLE_PATH = CACHE_DIR_PATH + "/index.pkl"
DOCMAP_PICKLE_PATH = CACHE_DIR_PATH + "/docmap.pkl"
TERM_FREQUENCIES_PICKLE_PATH = CACHE_DIR_PATH + "/term_frequencies.pkl"
DATA_DIR_PATH = "data"
MOVIE_DATA_PATH = DATA_DIR_PATH + "/movies.json"
STOPWORDS_DATA_PATH = DATA_DIR_PATH + "/stopwords.txt"
MAX_RESULT = 5

# Load movies data
with open(MOVIE_DATA_PATH) as f:
    json_data = json.loads(f.read())

# Load stopwords
with open(STOPWORDS_DATA_PATH) as f:
    stopword_data = f.read()

movies: list[dict[str, Any]] = json_data.get("movies", [])
stopwords = set(stopword_data.splitlines())
punctuation_map = str.maketrans({p: "" for p in string.punctuation})
stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        # Tokenize the input
        tokens = tokenize_text(text)

        # Add each token to the index with the document ID
        for token in tokens:
            self.index.setdefault(token, set())
            self.index[token].add(doc_id)

            self.term_frequencies.setdefault(doc_id, Counter())
            self.term_frequencies[doc_id].update([token])

    @staticmethod
    def __tokenize_term(term: str) -> str:
        # Tokenize the input
        tokens = tokenize_text(term)
        if not tokens:
            raise Exception("Must have at least one token")
        if len(tokens) > 1:
            raise Exception("Cannot have more than one token")

        return tokens[0]

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(self.__tokenize_term(term))
        if doc_ids is None:
            return []

        result = list(doc_ids)
        result.sort
        return result

    def get_tf(self, doc_id: int, term: str) -> int:
        freq = self.term_frequencies[doc_id].get(self.__tokenize_term(term))
        if freq is None:
            return 0

        return freq

    def build(self) -> None:
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

    def save(self) -> None:
        os.makedirs(CACHE_DIR_PATH, exist_ok=True)

        with open(INDEX_PICKLE_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_PICKLE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TERM_FREQUENCIES_PICKLE_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)


def tokenize_text(text: str) -> list[str]:
    return [
        # Stemming
        stemmer.stem(cleaned, False)
        for token in text.lower().split()
        # Remove stopwords
        if token not in stopwords
        # Remove punctuation
        if (cleaned := token.translate(punctuation_map))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # Build
    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    # Term frequencies
    tf_parser = subparsers.add_parser("tf", help="Show the frequencies of given term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    args = parser.parse_args()
    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print(e)
                return

            # Tokenization query
            query_tokens = tokenize_text(args.query)
            results = []

            for token in query_tokens:
                doc_ids = inverted_index.get_documents(token)
                for doc in (inverted_index.docmap[doc_id] for doc_id in doc_ids):
                    results.append(doc)

            for result in results:
                print(f"Movie Title {result['title']}, Movie ID {result['id']}")

        case "build":
            inverted_index.build()
            inverted_index.save()

        case "tf":
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print(e)
                return

            freq = inverted_index.get_tf(args.doc_id, args.term)
            print(freq)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
