import argparse

from lib.multimodal_search import verify_image_embeddings


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

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_image_embeddings(args.image)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
