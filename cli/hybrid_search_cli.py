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
        "weighted-search", help="Search movies using weighted hybrid search"
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
        "rrf-search", help="Search movies using RRF hybrid search"
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
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = list(normalize(args.scores))
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, v in enumerate(results):
                print(f"\n{i + 1}. {v['doc']['title']}")
                print(f"(Hybrid Score: {v['hybrid_score']:.4f})")
                print(f"BM25: {v['bm25']:.4f}, Semantic: {v['semantic']:.4f}")
                print(f"{v['doc']['description'][:100]}...")

        case "rrf-search":
            results = rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )
            for i, v in enumerate(results):
                print(f"\n{i + 1}. {v['doc']['title']}")
                print(f"Rerank Score: {v.get('rerank_score', 0):.4f}")
                print(f"Batch Rerank Score: {v.get('batch_rerank_score', 0):.4f}")
                print(f"Cross Encoder Score: {v.get('cross_encoder_score', 0):.4f}")
                print(f"RRF Score: {v['rrf_score']:.4f}")
                print(f"BM25 Rank: {v['bm25']:.4f}, Semantic Rank: {v['semantic']:.4f}")
                print(f"{v['doc']['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
