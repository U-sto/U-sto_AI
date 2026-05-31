import os
import sys
import unittest
from collections import Counter


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AI_RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT_DIR, AI_RAG_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.build_chain_eval_dataset import build_dataset
from evaluation.metrics import compute_metrics


class TestEvaluationMetrics(unittest.TestCase):
    def test_compute_metrics_includes_recall_mrr_ndcg_and_context_precision(self):
        sample = {
            "question": "반납 절차 알려줘",
            "category": "물품 반납 관리",
            "source": "manual_chapter3.json",
            "chapter": "3.3.1",
            "title": "신규 물품 반납 등록 절차",
            "expected_abstain": False,
        }
        diagnostics = {
            "classification": "NEED_RAG",
            "reranked_scores": [
                {"source": "manual_chapter4.json", "chapter": "4.3.1", "category": "물품 불용 관리"},
                {
                    "source": "manual_chapter3.json",
                    "chapter": "3.3.1",
                    "category": "물품 반납 관리",
                    "title": "신규 물품 반납 등록 절차",
                },
            ],
            "final_context_scores": [
                {
                    "source": "manual_chapter3.json",
                    "chapter": "3.3.1",
                    "category": "물품 반납 관리",
                    "title": "신규 물품 반납 등록 절차",
                },
                {"source": "manual_chapter4.json", "chapter": "4.3.1", "category": "물품 불용 관리"},
            ],
            "final_context_count": 2,
        }

        metrics = compute_metrics(sample, diagnostics, "반납 절차 답변")

        self.assertEqual(metrics["recall_at_1"], 0)
        self.assertEqual(metrics["recall_at_3"], 1)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertGreater(metrics["ndcg_at_3"], 0)
        self.assertEqual(metrics["context_precision"], 0.5)
        self.assertEqual(metrics["abstention_correct"], 1)

    def test_compute_metrics_scores_expected_abstention(self):
        sample = {
            "question": "교내 식당 메뉴 알려줘",
            "expected_abstain": True,
            "expected_answerable": False,
        }
        diagnostics = {
            "classification": "NEED_RAG",
            "reranked_scores": [],
            "final_context_scores": [],
            "final_context_count": 0,
        }

        metrics = compute_metrics(sample, diagnostics, "죄송합니다, 매뉴얼에 해당 내용이 없어 답변드리기 어렵습니다.")

        self.assertTrue(metrics["abstained"])
        self.assertEqual(metrics["abstention_correct"], 1)

    def test_build_dataset_balances_categories_and_adds_abstention_cases(self):
        source_items = [
            {
                "question": "물품 취득 관리가 뭐야?",
                "answer": "취득 답변",
                "category": "물품 취득 관리",
                "source": "manual_chapter1.json",
                "chapter": "1.1",
                "title": "물품 취득 개요",
            },
            {
                "question": "신규 물품 반납 등록 절차를 알려줘.",
                "answer": "반납 답변",
                "category": "물품 반납 관리",
                "source": "manual_chapter3.json",
                "chapter": "3.3.1",
                "title": "신규 물품 반납 등록 절차",
            },
        ]

        rows = build_dataset(source_items, min_per_category=3)
        counts = Counter(row["category"] for row in rows)

        self.assertEqual(counts["물품 취득 관리"], 3)
        self.assertEqual(counts["물품 반납 관리"], 3)
        self.assertEqual(counts["근거 부족"], 20)
        self.assertTrue(any(row["expected_abstain"] for row in rows))


if __name__ == "__main__":
    unittest.main()
