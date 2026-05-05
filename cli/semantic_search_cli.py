#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    chunk,
    embed_chunks,
    embed_query_text,
    embed_text,
    search,
    semantic_chunk,
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

    # Chunk
    chunk_parser = subparsers.add_parser("chunk", help="Chunking text")
    chunk_parser.add_argument("text", type=str, help="Text for chunking")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for the text",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Overlap between chunks",
    )

    # Semantic Chunk
    semantic_chunk_parse = subparsers.add_parser("semantic_chunk", help="Chunking text")
    semantic_chunk_parse.add_argument("text", type=str, help="Text for chunking")
    semantic_chunk_parse.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="Chunk size for the text",
    )
    semantic_chunk_parse.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Overlap between chunks",
    )

    # Embed Chunk
    subparsers.add_parser("embed_chunks", help="Embed chunks")

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

        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk(args.text, args.chunk_size, args.overlap)

        case "embed_chunks":
            embed_chunks()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
