#!/usr/bin/env python3

import argparse

from lib.keyword_serach import (
    build_command,
    idf_command,
    mb25_idf_command,
    mb25_tf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from lib.search_utils import BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # Build
    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    # Term frequencies
    tf_parser = subparsers.add_parser(
        "tf", help="Show the term frequency for the given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    # Inverse Document Frequency
    idf_parser = subparsers.add_parser(
        "idf", help="Show the inverse document frequency for the given term"
    )
    idf_parser.add_argument("term", type=str, help="Search term")

    # TF-IDF
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Show the TF-IDF score for the given document ID and term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    # BM25 TF
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )

    # BM25 IDF
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    args = parser.parse_args()
    match args.command:
        case "search":
            results = search_command(args.query)
            for result in results:
                print(f"Movie Title {result['title']}, Movie ID {result['id']}")

        case "build":
            build_command()

        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(tf)

        case "idf":
            idf = idf_command(args.term)
            print("%.2f" % idf)

        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "bm25tf":
            bm25tf = mb25_tf_command(args.doc_id, args.term, args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25idf":
            bm25idf = mb25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
