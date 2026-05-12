import argparse

from lib.hybrid_search import rrf_search_command
from lib.search_utils import load_golden_dataset


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    for test_case in load_golden_dataset():
        results = rrf_search_command(test_case["query"], 60, limit)
        retrived_set = {result["doc"]["title"] for result in results}
        relevant_set = set(test_case["relevant_docs"])
        relevant_retrieved = len(retrived_set & relevant_set)

        precision = relevant_retrieved / len(retrived_set) if retrived_set else 0.0
        recall = relevant_retrieved / len(relevant_set) if relevant_set else 0.0

        print(f"Query: {test_case['query']}")
        print(f"Precision@{limit}: {precision:.4f}")
        print(f"Recall@{limit}: {recall:.4f}")
        print(f"Retrieved: {retrived_set}")
        print(f"Relevant: {relevant_set}")


if __name__ == "__main__":
    main()
