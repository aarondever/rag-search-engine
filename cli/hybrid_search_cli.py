import argparse

from lib.hybrid_search import normalize, rrf_search_command, weighted_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize
    normalize_parser = subparsers.add_parser("normalize", help="Normalize given scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores")

    # Weighted Search
    weighted_search_parser = subparsers.add_parser(
        "weighted_search", help="Search movies using weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, help="Alpha", default=0.5
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of query results",
    )

    # RRF Search
    rrf_parser = subparsers.add_parser(
        "rrf_search", help="Search movies using RRF hybrid search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, help="k", default=60)
    rrf_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of query results",
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = list(normalize(args.scores))
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted_search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, v in enumerate(results):
                print(f"\n{i + 1}. {v['doc']['title']}")
                print(f"(Hybrid Score: {v['hybrid_score']:.4f})")
                print(f"BM25: {v['bm25']:.4f}, Semantic: {v['semantic']:.4f}")
                print(f"{v['doc']['description'][:100]}...")

        case "rrf_search":
            results = rrf_search_command(args.query, args.k, args.limit, args.enhance)
            for i, v in enumerate(results):
                print(f"\n{i + 1}. {v['doc']['title']}")
                print(f"(RRF Score: {v['rrf_score']:.4f})")
                print(f"BM25: {v['bm25']:.4f}, Semantic: {v['semantic']:.4f}")
                print(f"{v['doc']['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
