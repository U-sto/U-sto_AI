from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


AI_RAG_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (AI_RAG_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app import config
from ingestion.embedder import get_embedding_model
from rag.chain import run_rag_chain
from vectorstore.chroma_store import load_chroma_db


def _load_samples(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {dataset_path}")

    samples = [
        {
            "eval_id": f"qa_{idx:03d}",
            **item,
        }
        for idx, item in enumerate(data)
        if isinstance(item, dict) and item.get("question")
    ]
    if limit is not None:
        samples = samples[:limit]
    return samples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "eval_id",
        "question",
        "category",
        "classification",
        "refined_query",
        "retrieved_count",
        "filtered_count",
        "final_context_count",
        "final_context_doc_types",
        "final_context_categories",
        "attribution_doc_ids",
        "answer_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            diagnostics = row.get("diagnostics") or {}
            attribution = row.get("attribution") or []
            writer.writerow(
                {
                    "eval_id": row.get("eval_id"),
                    "question": row.get("question"),
                    "category": row.get("category"),
                    "classification": diagnostics.get("classification"),
                    "refined_query": diagnostics.get("refined_query"),
                    "retrieved_count": diagnostics.get("retrieved_count"),
                    "filtered_count": diagnostics.get("filtered_count"),
                    "final_context_count": diagnostics.get("final_context_count"),
                    "final_context_doc_types": "|".join(diagnostics.get("final_context_doc_types") or []),
                    "final_context_categories": "|".join(diagnostics.get("final_context_categories") or []),
                    "attribution_doc_ids": "|".join(
                        str(item.get("doc_id")) for item in attribution if item.get("doc_id")
                    ),
                    "answer_preview": str(row.get("answer", ""))[:300],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the production RAG chain and save per-question diagnostics."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "qa_output" / "manual_qa_final.json",
    )
    parser.add_argument(
        "--vector-db-path",
        type=Path,
        default=PROJECT_ROOT / config.VECTOR_DB_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AI_RAG_DIR / "results" / "chain_diagnostics",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or environment variables.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(args.dataset_path, args.limit)
    embeddings = get_embedding_model()
    vectordb = load_chroma_db(embeddings=embeddings, persist_dir=str(args.vector_db_path))
    llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE)

    rows = []
    for index, sample in enumerate(samples, start=1):
        question = sample["question"]
        print(f"[{index}/{len(samples)}] {sample['eval_id']} {question[:60]}")
        result = run_rag_chain(llm, vectordb, question)
        rows.append(
            {
                "eval_id": sample["eval_id"],
                "question": question,
                "category": sample.get("category", ""),
                "reference_answer": sample.get("answer", ""),
                "answer": result.get("answer", ""),
                "attribution": result.get("attribution", []),
                "diagnostics": result.get("diagnostics", {}),
            }
        )

    jsonl_path = args.output_dir / "chain_diagnostics.jsonl"
    csv_path = args.output_dir / "chain_diagnostics.csv"
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
