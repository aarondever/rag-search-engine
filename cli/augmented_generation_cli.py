import argparse
import os

from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import rrf_search_command

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


client = genai.Client(api_key=api_key)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query, 60, 5)
            docs = chr(10).join(
                f"{r['doc'].get('title', '')} - {r['doc'].get('description', '')}"
                for r in results
            )

            print("Search Results:")
            for r in results:
                print(f"- {r['doc'].get('title', '')}")
            print()

            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=f"""You are a RAG agent for Hoopla, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                Provide a comprehensive answer that addresses the user's query.

                Query: {query}

                Documents:
                {docs}

                Answer:""",
            )

            print(f"RAG Response:\n{response.text}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
