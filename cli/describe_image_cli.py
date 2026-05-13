import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


client = genai.Client(api_key=api_key)


def main():
    parser = argparse.ArgumentParser(description="Multimodal CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search
    search_parser = subparsers.add_parser(
        "search", help="Search for movies based on a query and generate a response"
    )
    search_parser.add_argument(
        "--image", type=str, help="Path to the image file to describe"
    )
    search_parser.add_argument(
        "--query",
        type=str,
        help="Text query to provide context for the image description",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            image_path = args.image

            mime, _ = mimetypes.guess_type(image_path)
            mime = mime or "image/jpeg"

            with open(image_path, "rb") as f:
                img = f.read()

            system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary"""
            parts = [
                system_prompt,
                types.Part.from_bytes(data=img, mime_type=mime),
                args.query.strip(),
            ]
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=parts,
            )
            print(f"Rewritten query: {response.text.strip()}")
            if response.usage_metadata is not None:
                print(f"Total tokens:    {response.usage_metadata.total_token_count}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
