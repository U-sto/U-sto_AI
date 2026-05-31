from __future__ import annotations

import math
from typing import Any


METRIC_K_VALUES = (1, 3, 5, 10)
ABSTENTION_MARKERS = (
    "죄송합니다, 매뉴얼에 해당 내용이 없어 답변드리기 어렵습니다.",
    "답변드리기 어렵",
    "문서가 부족",
    "근거가 부족",
    "확인할 수 없습니다",
    "매뉴얼에 해당 내용이 없어",
)


def _expected_field(sample: dict[str, Any], *names: str) -> str:
    for name in names:
        value = sample.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def relevance_grade(doc: dict[str, Any], sample: dict[str, Any]) -> int:
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
    if source_matches and not expected_chapter:
        return 2
    if category_matches:
        return 1
    return 0


def first_relevant_rank(docs: list[dict[str, Any]], sample: dict[str, Any]) -> int | None:
    for rank, doc in enumerate(docs, start=1):
        if relevance_grade(doc, sample) > 0:
            return rank
    return None


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def ndcg_at_k(docs: list[dict[str, Any]], sample: dict[str, Any], k: int) -> float:
    grades = [relevance_grade(doc, sample) for doc in docs[:k]]
    if not grades:
        return 0.0
    ideal = sorted(grades, reverse=True)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(grades) / ideal_dcg


def context_precision(contexts: list[dict[str, Any]], sample: dict[str, Any]) -> float:
    if not contexts:
        return 0.0
    relevant_count = sum(1 for context in contexts if relevance_grade(context, sample) > 0)
    return relevant_count / len(contexts)


def is_abstained(answer: str, diagnostics: dict[str, Any]) -> bool:
    if diagnostics.get("final_context_count") == 0 and diagnostics.get("classification") == "NEED_RAG":
        return True
    return any(marker and marker in answer for marker in ABSTENTION_MARKERS)


def compute_metrics(sample: dict[str, Any], diagnostics: dict[str, Any], answer: str) -> dict[str, Any]:
    ranked_docs = diagnostics.get("reranked_scores") or diagnostics.get("retrieved_scores") or []
    final_context = diagnostics.get("final_context_scores") or []
    expected_abstain = bool(sample.get("expected_abstain") or sample.get("expected_answerable") is False)
    abstained = is_abstained(answer, diagnostics)

    first_rank = first_relevant_rank(ranked_docs, sample)
    metrics: dict[str, Any] = {
        "first_relevant_rank": first_rank,
        "mrr": round(1 / first_rank, 6) if first_rank else 0.0,
        "context_precision": round(context_precision(final_context, sample), 6),
        "expected_abstain": expected_abstain,
        "abstained": abstained,
        "abstention_correct": int(abstained == expected_abstain),
    }
    for k in METRIC_K_VALUES:
        metrics[f"recall_at_{k}"] = int(first_rank is not None and first_rank <= k)
        metrics[f"ndcg_at_{k}"] = round(ndcg_at_k(ranked_docs, sample, k), 6)
    return metrics
