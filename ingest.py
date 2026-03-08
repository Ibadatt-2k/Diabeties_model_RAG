#!/usr/bin/env python3
"""
Ingestion script — load documents, chunk, embed, and store in ChromaDB.

Usage:
  python ingest.py
  python ingest.py --docs data/raw/ --collection docs
  python ingest.py --reset          # wipe existing collection before ingesting
"""
import argparse
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store
from src.ingestion.loader import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("--docs",       default="data/raw/",  help="Directory containing source documents")
    parser.add_argument("--collection", default="docs",       help="ChromaDB collection name")
    parser.add_argument("--reset",      action="store_true",  help="Wipe existing collection before ingesting (prevents duplicates)")
    args = parser.parse_args()

    docs_path = Path(args.docs)
    if not docs_path.exists():
        print(f"Error: '{args.docs}' does not exist. Place your documents there first.")
        return

    print("=" * 55)
    print("  RAG Ingestion Pipeline")
    print("=" * 55)
    print(f"  Source      : {args.docs}")
    print(f"  Collection  : {args.collection}")
    print(f"  Reset       : {'yes — wiping existing data first' if args.reset else 'no'}")
    print("=" * 55 + "\n")

    if args.reset:
        print("Resetting collection...")

    print("Step 1/3  Loading documents...")
    documents = load_documents(args.docs)
    if not documents:
        print("No supported documents found. Supported: .pdf .html .htm .md .txt")
        return

    print("\nStep 2/3  Chunking documents...")
    chunks = chunk_documents(documents)

    print("\nStep 3/3  Embedding and storing in ChromaDB...")
    embed_and_store(chunks, collection_name=args.collection, reset=args.reset)

    print(f"\n  Ingestion complete — {len(chunks)} chunks stored in '{args.collection}'")


if __name__ == "__main__":
    main()
