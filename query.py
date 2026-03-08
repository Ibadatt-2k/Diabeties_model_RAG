#!/usr/bin/env python3
"""
Query script — ask a question against the ingested documentation.

Usage:
  python query.py "What are the recommended first-line treatments for hypertension?"
  python query.py "What is the dosage of metformin for type 2 diabetes?" --phase 1
  python query.py "What does the WHO guideline say about antibiotic resistance?" --phase 2
"""
import argparse

from src.pipeline import RAGPipeline


def _print_result(result: dict) -> None:
    print("\n" + "=" * 55)
    print("  ANSWER")
    print("=" * 55)
    print(result["answer"])

    if result.get("sources"):
        print("\n" + "-" * 55)
        print("  SOURCES")
        print("-" * 55)
        for i, src in enumerate(result["sources"], start=1):
            print(f"  {i}. {src['source_file']}  chunk #{src['chunk_index']}  score={src['score']:.4f}")
            print(f"     {src['preview'][:120]}...")

    if result.get("declined"):
        print("\n  [System declined — insufficient evidence in retrieved chunks]")

    if pv := result.get("prompt_version"):
        print(f"\n  [Prompt version: {pv}  |  Pipeline phase: {result.get('phase')}]")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question",     help="The question to ask")
    parser.add_argument("--phase",      type=int, default=2, choices=[1, 2],
                        help="1=basic vector search, 2=hybrid+rerank+citation guard (default)")
    parser.add_argument("--collection", default="docs", help="ChromaDB collection name")
    args = parser.parse_args()

    print(f"\n  Question : {args.question}")
    print(f"  Phase    : {args.phase}")

    pipeline = RAGPipeline(collection_name=args.collection, phase=args.phase)
    result   = pipeline.query(args.question)
    _print_result(result)


if __name__ == "__main__":
    main()
