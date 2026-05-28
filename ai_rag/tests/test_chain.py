import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag import chain
from rag.chain import run_rag_chain


def _doc(
    doc_id: str,
    content: str = "물품 반납 절차 안내",
    category: str = "물품 반납 관리",
    doc_type: str = "qa",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "source": "manual.json",
            "chapter": "반납",
            "category": category,
            "title": "반납 절차",
            "doc_type": doc_type,
        },
    )


class TestRunRagChain(unittest.TestCase):
    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["BACKEND_API_URL"] = "http://test-backend"
        os.environ["FRONTEND_BASE_URL"] = "http://test-frontend"

        self.base_llm = MagicMock(name="base_llm")
        self.tool_llm = MagicMock(name="tool_llm")
        self.base_llm.bind_tools.return_value = self.tool_llm
        self.tool_llm.invoke.return_value = AIMessage(content="", tool_calls=[])

    def test_rag_flow_uses_refined_query_and_returns_diagnostics(self):
        vectordb = MagicMock(name="vectordb")
        docs = [(_doc("doc-1"), 0.11), (_doc("doc-2"), 0.18)]
        self.base_llm.invoke.return_value = AIMessage(content="반납 절차 답변")

        with (
            patch.object(chain, "USE_RERANKING", False),
            patch.object(chain, "classify_question", return_value="NEED_RAG"),
            patch.object(chain, "refine_query", return_value="물품 반납 절차"),
            patch.object(chain, "retrieve_candidate_docs", return_value=docs) as retrieve_mock,
            patch.object(chain, "filter_retrieved_docs", return_value=docs),
        ):
            result = run_rag_chain(self.base_llm, vectordb, "반납은 어떻게 해?")

        retrieve_mock.assert_called_once_with(
            vectordb=vectordb,
            user_query="반납은 어떻게 해?",
            refined_query="물품 반납 절차",
            top_k=chain.RETRIEVER_TOP_K,
        )
        self.assertEqual(result["answer"], "반납 절차 답변")
        self.assertEqual(result["diagnostics"]["classification"], "NEED_RAG")
        self.assertEqual(result["diagnostics"]["refined_query"], "물품 반납 절차")
        self.assertEqual(result["diagnostics"]["final_context_count"], 2)
        self.assertEqual(result["diagnostics"]["final_context_doc_types"], ["qa", "qa"])
        self.assertEqual(result["attribution"][0]["doc_id"], "doc-1")

    def test_rag_flow_returns_no_context_when_filter_removes_all(self):
        docs = [(_doc("doc-1"), 9.9)]

        with (
            patch.object(chain, "classify_question", return_value="NEED_RAG"),
            patch.object(chain, "refine_query", return_value="불용 절차"),
            patch.object(chain, "retrieve_candidate_docs", return_value=docs),
            patch.object(chain, "filter_retrieved_docs", return_value=[]),
        ):
            result = run_rag_chain(self.base_llm, MagicMock(), "불용은 어떻게 해?")

        self.assertEqual(result["answer"], chain.NO_CONTEXT_RESPONSE)
        self.assertEqual(result["diagnostics"]["filtered_count"], 0)

    def test_no_rag_question_skips_retrieval(self):
        self.base_llm.invoke.return_value = AIMessage(content="안녕하세요.")

        with (
            patch.object(chain, "classify_question", return_value="NO_RAG"),
            patch.object(chain, "retrieve_candidate_docs") as retrieve_mock,
        ):
            result = run_rag_chain(self.base_llm, MagicMock(), "안녕")

        retrieve_mock.assert_not_called()
        self.assertEqual(result["answer"], "안녕하세요.")
        self.assertEqual(result["attribution"], [])

    def test_tool_result_skips_rag(self):
        fake_tool = MagicMock()
        fake_tool.invoke.return_value = json.dumps({"result": "재고 10대 있음"}, ensure_ascii=False)
        self.tool_llm.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_item_detail_info",
                    "args": {"asset_name": "노트북"},
                    "id": "call-1",
                }
            ],
        )
        self.base_llm.invoke.return_value = AIMessage(content="노트북 재고는 10대입니다.")

        with (
            patch.object(chain, "TOOL_MAP", {"get_item_detail_info": fake_tool}),
            patch.object(chain, "classify_question") as classify_mock,
        ):
            result = run_rag_chain(self.base_llm, MagicMock(), "노트북 재고 조회해줘")

        fake_tool.invoke.assert_called_once_with({"asset_name": "노트북"})
        classify_mock.assert_not_called()
        self.assertEqual(result["answer"], "노트북 재고는 10대입니다.")
        self.assertEqual(result["attribution"], [])

    def test_filter_retrieved_docs_uses_adaptive_cutoff_when_config_threshold_is_wide(self):
        docs = [
            (_doc("near"), 0.10),
            (_doc("still-near"), 0.18),
            (_doc("far"), 0.80),
        ]

        filtered = chain.filter_retrieved_docs(docs, threshold=10.0)

        self.assertEqual([doc.metadata["doc_id"] for doc, _score in filtered], ["near", "still-near"])

    def test_classify_question_defaults_to_need_rag_on_invalid_output(self):
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "잘 모르겠습니다"

        with patch.object(chain, "_build_text_chain", return_value=fake_chain):
            result = chain.classify_question(MagicMock(), "반납 절차 알려줘")

        self.assertEqual(result, "NEED_RAG")

    def test_context_sort_prefers_manual_and_faq_over_qa(self):
        docs = [
            _doc("qa-1", doc_type="qa"),
            _doc("manual-1", doc_type="manual_chunk"),
            _doc("faq-1", doc_type="faq"),
        ]

        sorted_docs = chain._sort_docs_for_context(docs)

        self.assertEqual([doc.metadata["doc_id"] for doc in sorted_docs], ["manual-1", "faq-1", "qa-1"])

    def test_context_focus_keeps_dominant_category_for_non_comparison_question(self):
        docs = [
            _doc("return-1", category="물품 반납 관리"),
            _doc("return-2", category="물품 반납 관리"),
            _doc("acquire-1", category="물품 취득 관리"),
        ]

        focused = chain._focus_docs_by_category(docs, "반납 절차 알려줘")

        self.assertEqual([doc.metadata["doc_id"] for doc in focused], ["return-1", "return-2"])

    def test_context_focus_keeps_multiple_categories_for_comparison_question(self):
        docs = [
            _doc("return-1", category="물품 반납 관리"),
            _doc("return-2", category="물품 반납 관리"),
            _doc("disuse-1", category="물품 불용 관리"),
        ]

        focused = chain._focus_docs_by_category(docs, "반납과 불용 차이 알려줘")

        self.assertEqual([doc.metadata["doc_id"] for doc in focused], ["return-1", "return-2", "disuse-1"])


if __name__ == "__main__":
    unittest.main()
