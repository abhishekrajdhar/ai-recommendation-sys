from __future__ import annotations

# When this script is executed directly (python scripts/build_index.py), the
# project root is not automatically on sys.path which causes imports like
# `from app.services.retrieval import ...` to fail. Insert the project root
# at the front of sys.path so local package imports work either way.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.retrieval import HybridRetriever


def main() -> None:
    retriever = HybridRetriever()
    retriever.build_index()
    print(f"Indexed {len(retriever.assessments)} SHL assessments")


if __name__ == "__main__":
    main()
