#!/usr/bin/env python3
"""
RAGAS offline evaluation script.

Runs every question in the golden dataset through the Phase 2 pipeline,
then measures:
  - faithfulness       : are the claims in the answer supported by the chunks?
  - answer_relevancy   : does the answer actually address the question?
  - context_precision  : are the retrieved chunks relevant to the question?
  - context_recall     : did retrieval find the chunks needed to answer?

Usage:
  python -m src.evaluation.run_eval
  python -m src.evaluation.run_eval --collection docs
"""
import argparse
import json
import os
from pathlib import Path

DATASET_PATH = Path("data/eval/golden_dataset.json")
RESULTS_PATH = Path("data/eval/eval_results.json")


def run_evaluation(collection_name: str = "docs") -> dict:
    # Late imports so env is loaded first
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.pipeline import RAGPipeline

    # ── load golden dataset ───────────────────────────────────────────────────
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. "
            "Run `python -m src.evaluation.build_dataset` first."
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    print(f"Loaded {len(golden)} golden Q&A pairs.")

    # ── run pipeline on every question ────────────────────────────────────────
    pipeline = RAGPipeline(collection_name=collection_name, phase=2)

    questions, answers, contexts, ground_truths = [], [], [], []

    for entry in golden:
        q      = entry["question"]
        result = pipeline.query(q)

        questions.append(q)
        answers.append(result["answer"])
        # Full chunk content for RAGAS — not just the preview
        contexts.append([s["content"] for s in result.get("sources", [])])
        ground_truths.append(entry["ground_truth"])

        print(f"  [{entry['id']:>3}] {q[:65]}...")

    # ── build RAGAS dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # ── configure RAGAS to use Groq LLM + local embeddings ───────────────────
    groq_llm = LangchainLLMWrapper(ChatGroq(
        model        = "llama-3.3-70b-versatile",
        temperature  = 0,
        groq_api_key = os.getenv("GROQ_API_KEY"),
    ))
    local_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name   = "all-MiniLM-L6-v2",
        model_kwargs = {"device": "cpu"},
    ))

    # ── run RAGAS ─────────────────────────────────────────────────────────────
    print("\nRunning RAGAS evaluation (this calls the LLM for each metric)...")
    scores = evaluate(
        dataset    = dataset,
        metrics    = [faithfulness, answer_relevancy, context_precision, context_recall],
        llm        = groq_llm,
        embeddings = local_embeddings,
    )

    # ── print summary ─────────────────────────────────────────────────────────
    summary = {
        "faithfulness":      float(scores["faithfulness"]),
        "answer_relevancy":  float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall":    float(scores["context_recall"]),
    }

    print("\n" + "=" * 55)
    print("  RAGAS Evaluation Results")
    print("=" * 55)
    for metric, score in summary.items():
        bar   = "█" * int(score * 20)
        empty = "░" * (20 - int(score * 20))
        print(f"  {metric:<22} {score:.4f}  {bar}{empty}")
    print("=" * 55)

    # ── save detailed results ─────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary":      summary,
        "dataset_size": len(golden),
        "per_question": scores.to_pandas().to_dict(orient="records"),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Detailed results saved to {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="docs")
    args = parser.parse_args()
    run_evaluation(collection_name=args.collection)
