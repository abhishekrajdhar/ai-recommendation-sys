from __future__ import annotations

from app.services.retrieval import HybridRetriever


def main() -> None:
    retriever = HybridRetriever()
    retriever.build_index()
    print(f"Indexed {len(retriever.assessments)} SHL assessments")


if __name__ == "__main__":
    main()
