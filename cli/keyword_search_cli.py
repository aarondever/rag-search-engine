#!/usr/bin/env python3

import argparse
import json
import string
from typing import Any

from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
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

            # Tokenization query
            query_tokens = args.query.lower().split()
            query_tokens = [
                # Stemming
                stemmer.stem(cleaned, False)
                for token in query_tokens
                # Remove stopwords
                if token not in stopwords
                # Remove punctuation
                if (cleaned := token.translate(punctuation_map))
            ]
            results = []

            for movie in movies:
                # Tokenization movie title
                title_tokens = movie["title"].lower().split()
                title_tokens = (
                    # Stemming
                    stemmer.stem(cleaned, False)
                    for token in title_tokens
                    # Remove stopwords
                    if token not in stopwords
                    # Remove punctuation
                    if (cleaned := token.translate(punctuation_map))
                )

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
