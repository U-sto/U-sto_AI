import argparse
import csv
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_RAG_ROOT = PROJECT_ROOT / "ai_rag"
for path in (PROJECT_ROOT, AI_RAG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app.config as config
from ingestion.embedder import get_embedding_model
from langchain_openai import ChatOpenAI
from rag.chain import compare_retrieval_quality
from vectorstore.chroma_store import load_chroma_db


METRIC_K_VALUES = (1, 3, 5, 10)


def _parse_int_options(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_options(value: str, allowed: tuple[str, ...] | None = None) -> list[str]:
    options = [item.strip() for item in str(value).split(",") if item.strip()]
    if allowed is None:
        return options
    invalid = [option for option in options if option not in allowed]
    if invalid:
        raise ValueError(f"지원하지 않는 옵션입니다: {invalid}. allowed={allowed}")
    return options


def _parse_bool_options(value: str) -> list[bool]:
    mapping = {
        "on": True,
        "true": True,
        "1": True,
        "yes": True,
        "off": False,
        "false": False,
        "0": False,
        "no": False,
    }
    options = []
    for item in str(value).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"hybrid 옵션은 on/off 형식이어야 합니다: {item}")
        options.append(mapping[key])
    return options


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.json"))

    samples = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        for item in data:
            if isinstance(item, dict) and item.get("question"):
                samples.append(
                    {
                        "question": str(item["question"]).strip(),
                        "expected_doc_id": item.get("doc_id") or "",
                        "expected_source": item.get("source") or file_path.name,
                        "expected_chapter": item.get("chapter") or item.get("source_chapter") or "",
                        "expected_category": item.get("category") or "",
                        "expected_title": item.get("title") or item.get("source_title") or "",
                        "expected_abstain": bool(
                            item.get("expected_abstain") or item.get("expected_answerable") is False
                        ),
                    }
                )
            if limit and len(samples) >= limit:
                return samples
    return samples


def _relevance_grade(context: dict, sample: dict) -> int:
    """3=exact, 2=same source/chapter, 1=same category, 0=not relevant."""
    if sample["expected_doc_id"] and context.get("doc_id") == sample["expected_doc_id"]:
        return 3

    source_matches = (
        bool(sample["expected_source"])
        and context.get("source") == sample["expected_source"]
    )
    chapter_matches = (
        bool(sample["expected_chapter"])
        and context.get("chapter") == sample["expected_chapter"]
    )
    category_matches = (
        bool(sample["expected_category"])
        and context.get("category") == sample["expected_category"]
    )
    title_matches = (
        bool(sample["expected_title"])
        and context.get("title") == sample["expected_title"]
    )

    if source_matches and chapter_matches and (title_matches or not sample["expected_title"]):
        return 3
    if source_matches and chapter_matches:
        return 2
    if source_matches and not sample["expected_chapter"]:
        return 2
    if category_matches:
        return 1
    return 0


def _is_match(context: dict, sample: dict) -> bool:
    return _relevance_grade(context, sample) > 0


def _first_hit_rank(contexts: list[dict], sample: dict) -> int | None:
    for rank, context in enumerate(contexts, start=1):
        if _is_match(context, sample):
            return rank
    return None


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _ndcg_at_k(contexts: list[dict], sample: dict, k: int) -> float:
    grades = [_relevance_grade(context, sample) for context in contexts[:k]]
    if not grades:
        return 0.0
    ideal = sorted(grades, reverse=True)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(grades) / ideal_dcg


def _context_precision(contexts: list[dict], sample: dict) -> float:
    if not contexts:
        return 0.0
    relevant_count = sum(1 for context in contexts if _is_match(context, sample))
    return relevant_count / len(contexts)


def _diversity_violations(contexts: list[dict]) -> str:
    violations = []
    for field, limit in config.CONTEXT_DIVERSITY_LIMITS.items():
        counts = {}
        for context in contexts:
            value = str(context.get(field) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        over_values = [
            f"{field}:{value}={count}>{limit}"
            for value, count in sorted(counts.items())
            if count > int(limit)
        ]
        violations.extend(over_values)
    return "|".join(violations)


def main():
    parser = argparse.ArgumentParser(description="Compare original/refined/ensemble retrieval quality.")
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "dataset" / "qa_output"))
    parser.add_argument("--vector-db", default=str(PROJECT_ROOT / config.VECTOR_DB_PATH))
    parser.add_argument("--output", default=str(AI_RAG_ROOT / "results" / "retrieval_quality_compare.csv"))
    parser.add_argument("--markdown-output", default=str(AI_RAG_ROOT / "results" / "retrieval_quality_compare.md"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--use-llm-refine", action="store_true")
    parser.add_argument("--top-k-options", default=str(config.RETRIEVER_TOP_K))
    parser.add_argument("--threshold-strategies", default=",".join(config.RAG_THRESHOLD_STRATEGIES))
    parser.add_argument("--hybrid-options", default="on,off")
    args = parser.parse_args()

    samples = load_questions(Path(args.questions), limit=args.limit)
    if not samples:
        raise RuntimeError(f"평가 질문을 찾을 수 없습니다: {args.questions}")
    top_k_options = _parse_int_options(args.top_k_options)
    threshold_strategies = _parse_str_options(
        args.threshold_strategies,
        allowed=tuple(config.RAG_THRESHOLD_STRATEGIES),
    )
    hybrid_options = _parse_bool_options(args.hybrid_options)

    embeddings = get_embedding_model()
    vectordb = load_chroma_db(embeddings=embeddings, persist_dir=args.vector_db)
    llm = (
        ChatOpenAI(model_name=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE)
        if args.use_llm_refine
        else None
    )

    rows = []
    total_samples = len(samples)
    for question_no, sample in enumerate(samples, start=1):
        # 현재 몇 번째 질문인지 화면에 출력
        print(f"[{question_no}/{total_samples}] '{sample['question']}' 평가 진행 중... ")
        
        question = sample["question"]
        for top_k in top_k_options:
            for threshold_strategy in threshold_strategies:
                for use_hybrid in hybrid_options:
                    comparison_rows = compare_retrieval_quality(
                        vectordb,
                        question,
                        llm=llm,
                        retriever_top_k=top_k,
                        threshold_strategy=threshold_strategy,
                        use_hybrid=use_hybrid,
                    )
                    for row in comparison_rows:
                        final_context = row["final_context"]
                        first_hit_rank = _first_hit_rank(final_context, sample)
                        metric_values = {
                            f"recall_at_{k}": int(first_hit_rank is not None and first_hit_rank <= k)
                            for k in METRIC_K_VALUES
                        }
                        metric_values.update(
                            {
                                f"ndcg_at_{k}": round(_ndcg_at_k(final_context, sample, k), 6)
                                for k in METRIC_K_VALUES
                            }
                        )
                        rows.append(
                            {
                                "question_no": question_no,
                                "question": question,
                                "expected_doc_id": sample["expected_doc_id"],
                                "expected_source": sample["expected_source"],
                                "expected_chapter": sample["expected_chapter"],
                                "expected_category": sample["expected_category"],
                                "use_hybrid": row["use_hybrid"],
                                "retriever_top_k": row["retriever_top_k"],
                                "threshold_strategy": row["threshold_strategy"],
                                "similarity_threshold": row["similarity_threshold"],
                                "search_mode": row["search_mode"],
                                "refined_query": row.get("refined_query", ""),
                                "rerank_top_n": row["rerank_top_n"],
                                "retrieved_count": row["retrieved_count"],
                                "filtered_count": row["filtered_count"],
                                "final_context_count": row["final_context_count"],
                                "hit": int(first_hit_rank is not None),
                                "first_hit_rank": first_hit_rank or "",
                                "mrr": round(1 / first_hit_rank, 6) if first_hit_rank else 0,
                                **metric_values,
                                "context_precision": round(_context_precision(final_context, sample), 6),
                                "diversity_violations": _diversity_violations(final_context),
                                "final_doc_ids": "|".join(
                                    str(item.get("doc_id", "")) for item in final_context
                                ),
                                "final_context_json": json.dumps(final_context, ensure_ascii=False),
                            }
                        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        key = (
            row["use_hybrid"],
            row["retriever_top_k"],
            row["threshold_strategy"],
            row["search_mode"],
            row["rerank_top_n"],
        )
        summary.setdefault(
            key,
            {
                "hit": 0,
                "mrr": 0.0,
                "recall_at_5": 0.0,
                "ndcg_at_5": 0.0,
                "context_precision": 0.0,
                "total": 0,
                "violations": 0,
            },
        )
        summary[key]["hit"] += row["hit"]
        summary[key]["mrr"] += row["mrr"]
        summary[key]["recall_at_5"] += row["recall_at_5"]
        summary[key]["ndcg_at_5"] += row["ndcg_at_5"]
        summary[key]["context_precision"] += row["context_precision"]
        summary[key]["total"] += 1
        summary[key]["violations"] += int(bool(row["diversity_violations"]))

    print(f"Saved retrieval comparison: {output_path}")
    print(
        "use_hybrid, retriever_top_k, threshold_strategy, search_mode, rerank_top_n, hit_rate, recall@5, mean_mrr, "
        "mean_ndcg@5, context_precision, diversity_violation_rate"
    )
    markdown_lines = [
        "# Retrieval experiment comparison",
        "",
        "| hybrid | top_k | threshold | search_mode | rerank_top_n | hit_rate | recall@5 | mrr | ndcg@5 | context_precision | diversity_violation_rate |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (use_hybrid, top_k, threshold_strategy, search_mode, top_n), values in sorted(summary.items()):
        total = max(values["total"], 1)
        metrics = {
            "hit_rate": values["hit"] / total,
            "recall_at_5": values["recall_at_5"] / total,
            "mrr": values["mrr"] / total,
            "ndcg_at_5": values["ndcg_at_5"] / total,
            "context_precision": values["context_precision"] / total,
            "diversity_violation_rate": values["violations"] / total,
        }
        print(
            f"{use_hybrid}, {top_k}, {threshold_strategy}, {search_mode}, {top_n}, "
            f"{metrics['hit_rate']:.4f}, "
            f"{metrics['recall_at_5']:.4f}, "
            f"{metrics['mrr']:.4f}, "
            f"{metrics['ndcg_at_5']:.4f}, "
            f"{metrics['context_precision']:.4f}, "
            f"{metrics['diversity_violation_rate']:.4f}"
        )
        markdown_lines.append(
            "| {hybrid} | {top_k} | {threshold} | {search_mode} | {top_n} | {hit_rate:.4f} | {recall_at_5:.4f} | {mrr:.4f} | {ndcg_at_5:.4f} | {context_precision:.4f} | {diversity_violation_rate:.4f} |".format(
                hybrid="on" if use_hybrid else "off",
                top_k=top_k,
                threshold=threshold_strategy,
                search_mode=search_mode,
                top_n=top_n,
                **metrics,
            )
        )

    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Saved retrieval comparison summary: {markdown_path}")


if __name__ == "__main__":
    main()
