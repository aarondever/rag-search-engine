import argparse

from lib.hybrid_search import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize
    normalize_parser = subparsers.add_parser("normalize", help="Normalize given scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = list(normalize(args.scores))
            for score in scores:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
