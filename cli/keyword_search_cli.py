#!/usr/bin/env python3

import argparse
import json
import string
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    punctuation_map = str.maketrans({p: "" for p in string.punctuation})

    match args.command:
        case "search":
            # Load the movies data
            with open("data/movies.json") as f:
                json_data = json.loads(f.read())

            movies: list[dict[str, Any]] = json_data.get("movies", [])
            results = []

            for movie in movies:
                query = args.query.lower().translate(punctuation_map)
                title = movie["title"].lower().translate(punctuation_map)
                if query in title:
                    results.append(movie)

            # Truncate the list to a maximum of 5 results, order by IDs ascending.
            results = sorted(results, key=lambda x: x["id"])
            results = results[:5]

            for result in results:
                print(f"Movie Title {result['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
