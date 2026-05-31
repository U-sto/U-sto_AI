from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AI_RAG_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "qa_output" / "manual_qa_final.json"
DEFAULT_OUTPUT = AI_RAG_DIR / "evaluation" / "datasets" / "rag_chain_eval_dataset.json"
TARGET_MIN_PER_CATEGORY = 20

CONFUSION_FOCUS_CATEGORIES = {
    "물품 반납 관리",
    "물품 불용 관리",
    "물품 취득 관리",
    "물품 처분 관리",
    "사용주기 AI 예측",
}

QUESTION_VARIANTS = (
    ("source", "{question}"),
    ("paraphrase", "{question} 핵심만 정리해줘."),
    ("paraphrase", "매뉴얼 기준으로 {question}"),
    ("short_colloquial", "{question}"),
    ("short_colloquial", "{question} 짧게 알려줘."),
    ("short_colloquial", "{category}에서 {question}"),
    ("complex", "{question} 관련해서 조건이나 제약이 있으면 같이 알려줘."),
    ("complex", "{question} 버튼 흐름 중심으로 설명해줘."),
    ("complex", "{question} 사용자가 헷갈리지 않게 단계별로 알려줘."),
    ("complex", "{question} 관리자 처리까지 포함해서 알려줘."),
    ("complex", "{question} 등록/승인 관점으로 설명해줘."),
    ("complex", "{question} 조회 조건이 있으면 같이 알려줘."),
    ("complex", "{question} 취소나 수정 가능 조건도 있으면 알려줘."),
    ("paraphrase", "{category} 매뉴얼에 따르면 {question}"),
    ("complex", "{question} 실무자가 바로 따라 할 수 있게 정리해줘."),
    ("complex", "{question} 확정 전후 차이도 있으면 알려줘."),
    ("complex", "{question} 다른 절차와 헷갈리는 부분을 조심해서 알려줘."),
    ("paraphrase", "{category}에서 자주 묻는 질문인데, {question}"),
    ("complex", "{question} 필요한 입력 항목을 포함해서 알려줘."),
    ("complex", "{question} 결과 화면에는 무엇이 남는지도 알려줘."),
    ("complex", "{question} 한 문단 요약 후 세부 절차를 알려줘."),
)

CONFUSION_VARIANTS_BY_CATEGORY = {
    "물품 반납 관리": (
        "반납과 불용을 혼동하지 않도록 {question}",
        "반납, 불용, 처분 중 이 질문은 반납 절차 기준으로 {question}",
        "운용부서 반납과 불용 신청을 구분해서 {question}",
        "반납 확정과 불용 확정을 섞지 말고 {question}",
        "반납 후 이어지는 절차와 구분해서 {question}",
    ),
    "물품 불용 관리": (
        "불용과 반납을 혼동하지 않도록 {question}",
        "불용과 처분의 선후관계를 고려해서 {question}",
        "불용 신청과 처분 등록을 구분해서 {question}",
        "사용 중단 결정 절차라는 점을 반영해서 {question}",
        "불용 확정 전후 상태를 기준으로 {question}",
    ),
    "물품 취득 관리": (
        "취득과 운용 등록을 섞지 말고 {question}",
        "신규 물품 취득 기준으로 {question}",
        "취득일자와 정리일자 같은 취득 용어를 보존해서 {question}",
        "취득 승인요청과 확정 흐름을 구분해서 {question}",
        "G2B 목록 선택 단계와 일반 등록 단계를 나눠서 {question}",
    ),
    "물품 처분 관리": (
        "처분과 불용의 선후관계를 고려해서 {question}",
        "불용 확정 이후 처분 절차라는 점을 반영해서 {question}",
        "처분 등록과 불용 신청을 구분해서 {question}",
        "매각/폐기/양여 같은 처분 구분을 보존해서 {question}",
        "처분 확정 전후 상태를 기준으로 {question}",
    ),
    "사용주기 AI 예측": (
        "사용주기 AI 예측과 AI 챗봇 안내를 구분해서 {question}",
        "차트/지표 분석 기능 기준으로 {question}",
        "챗봇 답변이 아니라 예측 서비스 화면 기준으로 {question}",
        "정량 분석과 매뉴얼 안내를 섞지 말고 {question}",
        "사용주기 예측 결과 활용 관점에서 {question}",
    ),
}

UNANSWERABLE_SAMPLES = (
    "직원 연차 신청은 어디서 해?",
    "학생 장학금 신청 마감일을 알려줘.",
    "교내 식당 오늘 메뉴가 뭐야?",
    "주차 정기권 환불 절차를 알려줘.",
    "도서관 좌석 예약 취소는 어떻게 해?",
    "학사경고 기준을 알려줘.",
    "수강신청 정정 기간이 언제야?",
    "연구비 카드 분실 신고 방법 알려줘.",
    "교직원 급여 명세서는 어디서 확인해?",
    "메일 비밀번호 초기화 절차 알려줘.",
    "강의실 냉난방 고장 신고는 어떻게 해?",
    "졸업 요건 확인 방법 알려줘.",
    "출장비 정산 증빙 기준 알려줘.",
    "휴학 신청 승인 상태는 어디서 봐?",
    "기숙사 입사 신청 결과를 확인하고 싶어.",
    "교내 와이파이 인증 오류 해결해줘.",
    "회의실 예약 현황 조회 방법 알려줘.",
    "외부인 출입증 신청 절차 알려줘.",
    "증명서 발급 수수료가 얼마야?",
    "교내 셔틀버스 운행 시간을 알려줘.",
)


def _load_source(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return [item for item in data if isinstance(item, dict) and item.get("question")]


def _clean_question(question: str) -> str:
    return str(question).strip().rstrip(". ")


def _category_slug(category: str) -> str:
    mapping = {
        "물품 취득 관리": "acquisition",
        "물품 운용 관리": "operation",
        "절차": "operation",
        "식별": "identification",
        "물품 반납 관리": "return",
        "물품 불용 관리": "disuse",
        "물품 처분 관리": "disposal",
        "물품 보유 현황": "holding_status",
        "보유현황조회": "holding_status",
        "사용주기 AI 예측": "usage_prediction",
        "AI 챗봇": "ai_chatbot",
        "근거 부족": "abstention",
        "개념비교": "comparison",
        "일반": "general",
    }
    return mapping.get(category, "category")


def _make_answerable_sample(
    item: dict[str, Any],
    category: str,
    index: int,
) -> dict[str, Any]:
    base_question = _clean_question(item["question"])
    category_confusion_variants = CONFUSION_VARIANTS_BY_CATEGORY.get(category, ())
    if category_confusion_variants and index >= TARGET_MIN_PER_CATEGORY - len(category_confusion_variants):
        template = category_confusion_variants[index % len(category_confusion_variants)]
        question_type = "complex"
    else:
        variant_type, template = QUESTION_VARIANTS[index % len(QUESTION_VARIANTS)]
        question_type = variant_type

    question = template.format(question=base_question, category=category)
    return {
        "eval_id": f"{_category_slug(category)}_{index + 1:03d}",
        "question": question,
        "reference_question": item["question"],
        "answer": item.get("answer", ""),
        "category": category,
        "source": item.get("source", ""),
        "chapter": item.get("chapter", ""),
        "title": item.get("title", ""),
        "expected_source": item.get("source", ""),
        "expected_chapter": item.get("chapter", ""),
        "expected_category": category,
        "expected_title": item.get("title", ""),
        "expected_answerable": True,
        "expected_abstain": False,
        "question_type": question_type,
        "confusion_focus": category in CONFUSION_FOCUS_CATEGORIES,
    }


def build_dataset(source_items: list[dict[str, Any]], min_per_category: int) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source_items:
        by_category[str(item.get("category") or "unknown")].append(item)

    rows: list[dict[str, Any]] = []
    for category, items in sorted(by_category.items()):
        for index in range(min_per_category):
            item = items[index % len(items)]
            rows.append(_make_answerable_sample(item, category, index))

    for index, question in enumerate(UNANSWERABLE_SAMPLES, start=1):
        rows.append(
            {
                "eval_id": f"abstention_{index:03d}",
                "question": question,
                "reference_question": "",
                "answer": "",
                "category": "근거 부족",
                "source": "",
                "chapter": "",
                "title": "",
                "expected_source": "",
                "expected_chapter": "",
                "expected_category": "",
                "expected_title": "",
                "expected_answerable": False,
                "expected_abstain": True,
                "question_type": "evidence_gap",
                "confusion_focus": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a category-balanced end-to-end RAG evaluation dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-per-category", type=int, default=TARGET_MIN_PER_CATEGORY)
    args = parser.parse_args()

    source_items = _load_source(args.source)
    rows = build_dataset(source_items, args.min_per_category)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(row["category"] for row in rows)
    print(f"Wrote {len(rows)} samples to {args.output}")
    for category, count in sorted(counts.items()):
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
