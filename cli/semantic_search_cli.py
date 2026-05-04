#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify model
    subparsers.add_parser("verify", help="Verify loaded model")

    # Embed text
    embed_text_parser = subparsers.add_parser("embed_text", help="Embedding given text")
    embed_text_parser.add_argument("text", type=str, help="Text for embedding")

    # Verify embeddings
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    # Embed query
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query")
    embed_query_parser.add_argument("query", type=str, help="Search query")

    # Search
    search_parser = subparsers.add_parser(
        "search", help="Search movies using cosine similarity scoring"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit the number of query results",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
