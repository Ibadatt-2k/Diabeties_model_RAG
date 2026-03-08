#!/usr/bin/env python3
"""
Golden dataset curation tool.

Run this interactively to build your 50–200 verified Q&A pairs.
The pipeline answers each question; you verify and approve/correct before saving.

Usage:
  python -m src.evaluation.build_dataset
  python -m src.evaluation.build_dataset --collection docs
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = Path("data/eval/golden_dataset.json")


def _load() -> list:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(dataset: list) -> None:
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def add_entry(question: str, ground_truth: str, source_document: str) -> dict:
    dataset = _load()
    entry = {
        "id":              len(dataset) + 1,
        "question":        question,
        "ground_truth":    ground_truth,
        "source_document": source_document,
    }
    dataset.append(entry)
    _save(dataset)
    print(f"  Saved entry #{entry['id']}.")
    return entry


def interactive_curation(collection_name: str = "docs") -> None:
    """
    Interactive loop:
      1. You type a question.
      2. The pipeline answers it (phase 2).
      3. You decide: approve / edit / skip.
      4. Approved entries go into golden_dataset.json.
    """
    from src.pipeline import RAGPipeline   # late import so load_dotenv() runs first

    pipeline = RAGPipeline(collection_name=collection_name, phase=2)
    dataset  = _load()

    print("\n" + "=" * 55)
    print("  Golden Dataset Curation Tool")
    print(f"  Current dataset size: {len(dataset)} entries")
    print("  Type 'quit' to exit.")
    print("=" * 55 + "\n")

    while True:
        question = input("Question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break

        result = pipeline.query(question)
        print("\n  System answer:")
        print("  " + result["answer"].replace("\n", "\n  "))
        if result.get("sources"):
            print("  Sources:", [s["source_file"] for s in result["sources"]])

        action = input("\n  [a]pprove  [e]dit  [s]kip  > ").strip().lower()

        if action == "a":
            source = result["sources"][0]["source_file"] if result.get("sources") else "unknown"
            add_entry(question, result["answer"], source)

        elif action == "e":
            ground_truth = input("  Corrected ground truth answer:\n  > ").strip()
            source       = input("  Source document filename: ").strip()
            add_entry(question, ground_truth, source)

        updated = _load()
        print(f"\n  Dataset now has {len(updated)} entries.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="docs")
    args = parser.parse_args()
    interactive_curation(collection_name=args.collection)
