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

    # Rag
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of query results to include in the RAG answer",
    )

    # Summary
    summary_parser = subparsers.add_parser(
        "summary", help="Summarize RRF search results"
    )
    summary_parser.add_argument("query", type=str, help="Search query for summary")
    summary_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of query results",
    )

    # Citations
    citations_parser = subparsers.add_parser(
        "citations", help="Generate an answer with citations from search results"
    )
    citations_parser.add_argument(
        "query", type=str, help="Search query for answer with citations"
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of query results to include in the answer",
    )

    # Question
    question_parser = subparsers.add_parser(
        "question", help="Answer a user question based on search results"
    )
    question_parser.add_argument(
        "query", type=str, help="Search query for question answering"
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of query results to include in the answer",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            docs = retrieve_docs(args)
            content = generate_content(
                f"""You are a RAG agent for Hoopla, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                Provide a comprehensive answer that addresses the user's query.

                Query: {query}

                Documents:
                {docs}

                Answer:""",
            )

            print(f"RAG Response:\n{content}")

        case "summary":
            query = args.query
            docs = retrieve_docs(args)
            content = generate_content(
                f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Search results:
                {docs}

                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            )
            print(f"LLM Summary:\n{content}")

        case "citations":
            query = args.query
            docs = retrieve_docs(args)
            content = generate_content(
                f"""Answer the query below and give information based on the provided documents.

                The answer should be tailored to users of Hoopla, a movie streaming service.
                If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

                Query: {query}

                Documents:
                {docs}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources in the format [1], [2], etc. when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the provided documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""
            )
            print(f"LLM Answer:\n{content}")

        case "question":
            query = args.query
            docs = retrieve_docs(args)
            content = generate_content(
                f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

                Question: {query}

                Documents:
                {docs}

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation

                Answer:"""
            )
            print(f"Answer:\n{content}")

        case _:
            parser.print_help()


def retrieve_docs(args) -> str:
    query = args.query
    limit = args.limit
    results = rrf_search_command(query, 60, limit)
    docs = "\n".join(
        f"{r['doc'].get('title', '')} - {r['doc'].get('description', '')}"
        for r in results
    )

    print("Search Results:")
    for r in results:
        print(f"- {r['doc'].get('title', '')}")
    print()

    return docs


def generate_content(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    main()
