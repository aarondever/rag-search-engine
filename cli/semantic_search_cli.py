#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify model
    subparsers.add_parser("verify", help="Verify loaded model")

    # Embed text
    embed_text_parser = subparsers.add_parser("embed_text", help="Embedding given text")
    embed_text_parser.add_argument("text", type=str, help="Text for embedding")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
