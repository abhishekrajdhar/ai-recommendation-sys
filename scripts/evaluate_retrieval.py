import json
from pathlib import Path
from typing import List

from app.services.retrieval import HybridRetriever


def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    topk = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(topk) & set(relevant)) / len(set(relevant))


def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    topk = recommended[:k]
    if not topk:
        return 0.0
    return len(set(topk) & set(relevant)) / len(topk)


def mrr(recommended: List[str], relevant: List[str]) -> float:
    for i, r in enumerate(recommended, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    gt_path = root / "app" / "tests" / "ground_truth.json"

    with gt_path.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    retriever = HybridRetriever()

    ks = [1, 3, 5, 10]

    totals = {k: {"recall": 0.0, "precision": 0.0, "mrr": 0.0} for k in ks}

    for item in ground_truth:
        query = item["query"]
        relevant = item.get("relevant_urls", [])

        results = retriever.search(query, initial_k=20, limit=10)
        rec_urls = [r.url for r in results]

        print("Query:\t", query)
        print("Top results:")
        for u in rec_urls[:10]:
            print("  ", u)

        for k in ks:
            r = recall_at_k(rec_urls, relevant, k)
            p = precision_at_k(rec_urls, relevant, k)
            mm = mrr(rec_urls, relevant)

            totals[k]["recall"] += r
            totals[k]["precision"] += p
            totals[k]["mrr"] += mm

            print(f"  @ {k}: recall={r:.3f} precision={p:.3f} mrr={mm:.3f}")

        print("---")

    n = len(ground_truth)

    print("Average:")
    for k in ks:
        print(f"@{k}: recall={totals[k]['recall']/n:.3f} precision={totals[k]['precision']/n:.3f} mrr={totals[k]['mrr']/n:.3f}")
