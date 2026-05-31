from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
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
from evaluation.metrics import compute_metrics as compute_chain_metrics
from ingestion.embedder import get_embedding_model
from rag.chain import run_rag_chain
from vectorstore.chroma_store import load_chroma_db


METRIC_K_VALUES = (1, 3, 5, 10)
ABSTENTION_MARKERS = (
    config.NO_CONTEXT_RESPONSE,
    "답변드리기 어렵",
    "문서가 부족",
    "근거가 부족",
    "확인할 수 없습니다",
    "매뉴얼에 해당 내용이 없어",
)


def _load_samples(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {dataset_path}")

    samples = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("question"):
            continue

        eval_id = item.get("eval_id") or item.get("id") or f"qa_{idx:03d}"
        samples.append(
            {
                "eval_id": str(eval_id),
                **item,
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _doc_ids(docs: list[dict[str, Any]]) -> str:
    return "|".join(str(doc.get("doc_id", "")) for doc in docs if doc.get("doc_id"))


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "<br>").replace("\n", " ")


def _expected_field(sample: dict[str, Any], *names: str) -> str:
    for name in names:
        value = sample.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _relevance_grade(doc: dict[str, Any], sample: dict[str, Any]) -> int:
    """3=exact, 2=same source/chapter, 1=same category, 0=not relevant."""
    expected_doc_id = _expected_field(sample, "expected_doc_id", "doc_id")
    expected_source = _expected_field(sample, "expected_source", "source")
    expected_chapter = _expected_field(sample, "expected_chapter", "chapter", "source_chapter")
    expected_title = _expected_field(sample, "expected_title", "title", "source_title")
    expected_category = _expected_field(sample, "expected_category", "category")

    if expected_doc_id and str(doc.get("doc_id", "")) == expected_doc_id:
        return 3

    source_matches = bool(expected_source) and doc.get("source") == expected_source
    chapter_matches = bool(expected_chapter) and doc.get("chapter") == expected_chapter
    title_matches = bool(expected_title) and doc.get("title") == expected_title
    category_matches = bool(expected_category) and doc.get("category") == expected_category

    if source_matches and chapter_matches and (title_matches or not expected_title):
        return 3
    if source_matches and chapter_matches:
        return 2
    if category_matches:
        return 1
    return 0


def _first_relevant_rank(docs: list[dict[str, Any]], sample: dict[str, Any]) -> int | None:
    for rank, doc in enumerate(docs, start=1):
        if _relevance_grade(doc, sample) > 0:
            return rank
    return None


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _ndcg_at_k(docs: list[dict[str, Any]], sample: dict[str, Any], k: int) -> float:
    grades = [_relevance_grade(doc, sample) for doc in docs[:k]]
    if not grades:
        return 0.0
    ideal = sorted(grades, reverse=True)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(grades) / ideal_dcg


def _is_abstained(answer: str, diagnostics: dict[str, Any]) -> bool:
    if diagnostics.get("final_context_count") == 0 and diagnostics.get("classification") == "NEED_RAG":
        return True
    return any(marker and marker in answer for marker in ABSTENTION_MARKERS)


def _compute_metrics(sample: dict[str, Any], diagnostics: dict[str, Any], answer: str) -> dict[str, Any]:
    ranked_docs = diagnostics.get("reranked_scores") or diagnostics.get("retrieved_scores") or []
    final_context = diagnostics.get("final_context_scores") or []
    expected_abstain = bool(sample.get("expected_abstain") or sample.get("expected_answerable") is False)
    abstained = _is_abstained(answer, diagnostics)

    first_rank = _first_relevant_rank(ranked_docs, sample)
    metrics: dict[str, Any] = {
        "first_relevant_rank": first_rank,
        "mrr": round(1 / first_rank, 6) if first_rank else 0.0,
        "context_precision": round(
            sum(1 for doc in final_context if _relevance_grade(doc, sample) > 0) / len(final_context),
            6,
        )
        if final_context
        else 0.0,
        "expected_abstain": expected_abstain,
        "abstained": abstained,
        "abstention_correct": int(abstained == expected_abstain),
    }
    for k in METRIC_K_VALUES:
        metrics[f"recall_at_{k}"] = int(first_rank is not None and first_rank <= k)
        metrics[f"ndcg_at_{k}"] = round(_ndcg_at_k(ranked_docs, sample, k), 6)
    return metrics


def _diagnostic_record(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") or {}
    answer = result.get("answer", "")
    metrics = compute_chain_metrics(sample, diagnostics, answer)
    return {
        "eval_id": sample["eval_id"],
        "question": sample["question"],
        "category": sample.get("category", ""),
        "source": sample.get("source", ""),
        "chapter": sample.get("chapter", ""),
        "expected_doc_id": sample.get("expected_doc_id") or sample.get("doc_id") or "",
        "expected_answerable": sample.get("expected_answerable", not metrics["expected_abstain"]),
        "reference_answer": sample.get("answer") or sample.get("reference_answer", ""),
        "classification": diagnostics.get("classification"),
        "refined_query": diagnostics.get("refined_query"),
        "retrieved_docs": diagnostics.get("retrieved_scores", []),
        "filtered_docs": diagnostics.get("filtered_scores", []),
        "reranked_docs": diagnostics.get("reranked_scores", []),
        "final_context": diagnostics.get("final_context_scores", []),
        "final_context_text": diagnostics.get("final_context_text", ""),
        "answer": answer,
        "attribution": result.get("attribution", []),
        "diagnostics": diagnostics,
        "metrics": metrics,
    }


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
        "retrieved_doc_ids",
        "filtered_doc_ids",
        "reranked_doc_ids",
        "final_context_doc_ids",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
        "context_precision",
        "expected_abstain",
        "abstained",
        "abstention_correct",
        "attribution_doc_ids",
        "answer_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            diagnostics = row.get("diagnostics") or {}
            metrics = row.get("metrics") or {}
            attribution = row.get("attribution") or []
            retrieved_docs = row.get("retrieved_docs") or []
            filtered_docs = row.get("filtered_docs") or []
            reranked_docs = row.get("reranked_docs") or []
            final_context = row.get("final_context") or []
            writer.writerow(
                {
                    "eval_id": row.get("eval_id"),
                    "question": row.get("question"),
                    "category": row.get("category"),
                    "classification": row.get("classification"),
                    "refined_query": row.get("refined_query"),
                    "retrieved_count": diagnostics.get("retrieved_count", 0),
                    "filtered_count": diagnostics.get("filtered_count", 0),
                    "final_context_count": diagnostics.get("final_context_count", 0),
                    "retrieved_doc_ids": _doc_ids(retrieved_docs),
                    "filtered_doc_ids": _doc_ids(filtered_docs),
                    "reranked_doc_ids": _doc_ids(reranked_docs),
                    "final_context_doc_ids": _doc_ids(final_context),
                    "recall_at_1": metrics.get("recall_at_1", 0),
                    "recall_at_3": metrics.get("recall_at_3", 0),
                    "recall_at_5": metrics.get("recall_at_5", 0),
                    "recall_at_10": metrics.get("recall_at_10", 0),
                    "mrr": metrics.get("mrr", 0.0),
                    "ndcg_at_5": metrics.get("ndcg_at_5", 0.0),
                    "ndcg_at_10": metrics.get("ndcg_at_10", 0.0),
                    "context_precision": metrics.get("context_precision", 0.0),
                    "expected_abstain": metrics.get("expected_abstain", False),
                    "abstained": metrics.get("abstained", False),
                    "abstention_correct": metrics.get("abstention_correct", 0),
                    "attribution_doc_ids": "|".join(
                        str(item.get("doc_id")) for item in attribution if item.get("doc_id")
                    ),
                    "answer_preview": str(row.get("answer", ""))[:300],
                }
            )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _numeric_scores(docs: list[dict[str, Any]], field: str = "retrieval_score") -> list[float]:
    values = []
    for doc in docs:
        value = doc.get(field)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(math.ceil(len(ordered) * ratio) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p50": round(_percentile(values, 0.50) or 0.0, 6),
        "p90": round(_percentile(values, 0.90) or 0.0, 6),
        "max": round(max(values), 6),
        "mean": round(_mean(values), 6),
    }


def _score_distribution_by_retrieval_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {
        "answerable_with_relevant_doc_retrieved": [],
        "answerable_only_wrong_docs_retrieved": [],
        "unanswerable_retrieved": [],
    }
    for row in rows:
        retrieved_docs = row.get("retrieved_docs") or []
        scores = _numeric_scores(retrieved_docs)
        if not scores:
            continue

        best_score = min(scores)
        expected_abstain = bool((row.get("metrics") or {}).get("expected_abstain"))
        has_relevant_doc = _first_relevant_rank(retrieved_docs, row) is not None
        if expected_abstain:
            buckets["unanswerable_retrieved"].append(best_score)
        elif has_relevant_doc:
            buckets["answerable_with_relevant_doc_retrieved"].append(best_score)
        else:
            buckets["answerable_only_wrong_docs_retrieved"].append(best_score)

    return {name: _distribution(values) for name, values in buckets.items()}


def _failure_reason(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") or {}
    retrieved_docs = row.get("retrieved_docs") or []
    filtered_docs = row.get("filtered_docs") or []
    final_context = row.get("final_context") or []
    expected_abstain = bool(metrics.get("expected_abstain"))

    if expected_abstain and not metrics.get("abstained"):
        return "expected_abstain_but_answered"
    if not expected_abstain and metrics.get("abstained"):
        return "answerable_but_abstained"
    if not expected_abstain and _first_relevant_rank(retrieved_docs, row) is None:
        return "expected_doc_not_retrieved"
    if not expected_abstain and filtered_docs and _first_relevant_rank(filtered_docs, row) is None:
        return "expected_doc_removed_by_filter"
    if not expected_abstain and final_context and _first_relevant_rank(final_context, row) is None:
        return "expected_doc_missing_from_final_context"
    if not expected_abstain and float(metrics.get("context_precision", 0.0)) < 1.0:
        return "final_context_contains_irrelevant_docs"
    return ""


def _write_failure_reports(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    failures = [
        {
            **row,
            "failure_reason": reason,
        }
        for row in rows
        if (reason := _failure_reason(row))
    ]

    csv_path = output_dir / "chain_failure_cases.csv"
    md_path = output_dir / "chain_failure_cases.md"
    fieldnames = [
        "eval_id",
        "question",
        "category",
        "failure_reason",
        "expected_doc_id",
        "retrieved_doc_ids",
        "filtered_doc_ids",
        "reranked_doc_ids",
        "final_context_doc_ids",
        "answer_preview",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failures:
            writer.writerow(
                {
                    "eval_id": row.get("eval_id"),
                    "question": row.get("question"),
                    "category": row.get("category"),
                    "failure_reason": row.get("failure_reason"),
                    "expected_doc_id": row.get("expected_doc_id"),
                    "retrieved_doc_ids": _doc_ids(row.get("retrieved_docs") or []),
                    "filtered_doc_ids": _doc_ids(row.get("filtered_docs") or []),
                    "reranked_doc_ids": _doc_ids(row.get("reranked_docs") or []),
                    "final_context_doc_ids": _doc_ids(row.get("final_context") or []),
                    "answer_preview": str(row.get("answer", ""))[:300],
                }
            )

    lines = [
        "# Production chain failure cases",
        "",
        f"- Failure count: {len(failures)} / {len(rows)}",
        "",
        "| eval_id | category | reason | retrieved | filtered | final_context |",
        "|---|---|---|---|---|---|",
    ]
    for row in failures[:100]:
        lines.append(
            "| {eval_id} | {category} | {reason} | {retrieved} | {filtered} | {final_context} |".format(
                eval_id=_markdown_cell(row.get("eval_id", "")),
                category=_markdown_cell(row.get("category", "")),
                reason=_markdown_cell(row.get("failure_reason", "")),
                retrieved=_markdown_cell(_doc_ids(row.get("retrieved_docs") or [])),
                filtered=_markdown_cell(_doc_ids(row.get("filtered_docs") or [])),
                final_context=_markdown_cell(_doc_ids(row.get("final_context") or [])),
            )
        )
    if len(failures) > 100:
        lines.append("")
        lines.append(f"Only first 100 failures are shown. See `{csv_path.name}` for all rows.")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_names = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
        "context_precision",
        "abstention_correct",
    ]
    overall = {
        name: round(_mean([float((row.get("metrics") or {}).get(name, 0.0)) for row in rows]), 6)
        for name in metric_names
    }
    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category") or "unknown")].append(row)
    for category, items in sorted(grouped.items()):
        by_category[category] = {
            "sample_count": len(items),
            **{
                name: round(_mean([float((row.get("metrics") or {}).get(name, 0.0)) for row in items]), 6)
                for name in metric_names
            },
        }

    summary = {
        "evaluation_mode": "production_run_rag_chain",
        "sample_count": len(rows),
        "overall": overall,
        "by_category": by_category,
        "score_distribution_by_retrieval_outcome": _score_distribution_by_retrieval_outcome(rows),
        "metric_k_values": list(METRIC_K_VALUES),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]], summary_json_path: Path) -> None:
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    overall = summary["overall"]
    lines = [
        "# Production RAG chain diagnostics",
        "",
        "This report is generated from `run_rag_chain()` and is separate from the standalone LLM judge retrieval loop.",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Recall@5: {overall.get('recall_at_5', 0.0):.4f}",
        f"- MRR: {overall.get('mrr', 0.0):.4f}",
        f"- nDCG@5: {overall.get('ndcg_at_5', 0.0):.4f}",
        f"- Context precision: {overall.get('context_precision', 0.0):.4f}",
        f"- Abstention accuracy: {overall.get('abstention_correct', 0.0):.4f}",
        "",
        "## Score Distribution",
        "",
        "| bucket | count | min | p50 | p90 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, values in summary["score_distribution_by_retrieval_outcome"].items():
        lines.append(
            "| {bucket} | {count} | {min} | {p50} | {p90} | {max} | {mean} |".format(
                bucket=bucket,
                count=values.get("count", 0),
                min=values.get("min", ""),
                p50=values.get("p50", ""),
                p90=values.get("p90", ""),
                max=values.get("max", ""),
                mean=values.get("mean", ""),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the production RAG chain and save end-to-end diagnostics plus retrieval metrics."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=AI_RAG_DIR / "evaluation" / "datasets" / "rag_chain_eval_dataset.json",
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
    parser.add_argument("--limit", type=int, default=None)
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
        rows.append(_diagnostic_record(sample, result))

    jsonl_path = args.output_dir / "chain_diagnostics.jsonl"
    csv_path = args.output_dir / "chain_diagnostics.csv"
    summary_path = args.output_dir / "chain_diagnostics_summary.json"
    summary_md_path = args.output_dir / "chain_diagnostics_summary.md"
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)
    _write_summary(summary_path, rows)
    _write_summary_markdown(summary_md_path, rows, summary_path)
    failure_csv_path, failure_md_path = _write_failure_reports(args.output_dir, rows)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {failure_csv_path}")
    print(f"Wrote {failure_md_path}")


if __name__ == "__main__":
    main()
