#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import string
from typing import Any

from nltk.stem import PorterStemmer

# Load movies data
with open("data/movies.json") as f:
    json_data = json.loads(f.read())

# Load stopwords
with open("data/stopwords.txt") as f:
    stopword_data = f.read()

movies: list[dict[str, Any]] = json_data.get("movies", [])
stopwords = set(stopword_data.splitlines())
punctuation_map = str.maketrans({p: "" for p in string.punctuation})
stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, Any]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        # Tokenize the input text
        tokens = tokenize_text(text)

        # Add each token to the index with the document ID
        for token in tokens:
            self.index.setdefault(token, set())
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower())
        if doc_ids is None:
            return []

        result = list(doc_ids)
        result.sort
        return result

    def build(self) -> None:
        for movie in movies:
            doc_id = movie["id"]
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)


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

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            # Tokenization query
            query_tokens = tokenize_text(args.query)
            results = []

            for movie in movies:
                # Tokenization movie title
                title_tokens = tokenize_text(movie["title"])

                # Matching query with title tokens
                for title in title_tokens:
                    for query in query_tokens:
                        if query in title:
                            results.append(movie)
                            break

            for result in results:
                print(f"Movie Title {result['title']}")

            # Truncate the list to a maximum of 5 results, order by IDs ascending.
            results = sorted(results, key=lambda x: x["id"])
            results = results[:5]
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

            docs = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
