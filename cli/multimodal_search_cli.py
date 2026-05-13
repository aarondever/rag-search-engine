import argparse

from lib.multimodal_search import search_with_image, verify_image_embeddings


def main():
    parser = argparse.ArgumentParser(description="Multimodal CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify image embedding
    verify_parser = subparsers.add_parser(
        "verify", help="Verify the embeddings of an image"
    )
    verify_parser.add_argument(
        "--image", type=str, help="Path to the image file to verify"
    )

    # search with image
    search_parser = subparsers.add_parser(
        "search", help="Search for movies using an image"
    )
    search_parser.add_argument(
        "--image", type=str, help="Path to the image file to search with"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_image_embeddings(args.image)

        case "search":
            results = search_with_image(args.image)
            for i, result in enumerate(results):
                print(
                    f"{i + 1}. {result['title']} (Similarity: {result['similarity']:.4f})"
                )
                print(result["description"], "\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
