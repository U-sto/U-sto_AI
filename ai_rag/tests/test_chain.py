import json
import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag import chain
from rag.chain import run_rag_chain


def _doc(
    doc_id: str,
    content: str = "물품 반납 절차 안내",
    category: str = "물품 반납 관리",
    doc_type: str = "manual_chunk",
    chapter: str = "반납",
    source: str = "manual.json",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "source": source,
            "chapter": chapter,
            "category": category,
            "title": f"{category} 절차",
            "section_path": f"{chapter} > {category}",
            "chunk_index": 1,
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
        self.base_llm.invoke.return_value = AIMessage(content="테스트 답변")

        self.log_patcher = patch.object(chain, "_write_retrieval_log")
        self.log_patcher.start()

    def tearDown(self):
        self.log_patcher.stop()

    def test_procedure_question_routes_to_rag(self):
        vectordb = MagicMock(name="vectordb")
        docs = [(_doc("return-1"), 0.11), (_doc("return-2"), 0.18)]
        self.base_llm.invoke.return_value = AIMessage(content="반납 절차 답변")

        with (
            patch.object(chain, "USE_RERANKING", False),
            patch.object(chain, "classify_question", return_value="NEED_RAG") as classify_mock,
            patch.object(chain, "refine_query", return_value="물품 반납 절차") as refine_mock,
            patch.object(chain, "retrieve_candidate_docs", return_value=docs) as retrieve_mock,
            patch.object(chain, "filter_retrieved_docs", return_value=docs),
        ):
            result = run_rag_chain(self.base_llm, vectordb, "반납 절차 알려줘")

        self.base_llm.bind_tools.assert_called_once_with(chain.TOOLS)
        classify_mock.assert_called_once_with(self.base_llm, "반납 절차 알려줘")
        refine_mock.assert_called_once_with(self.base_llm, "반납 절차 알려줘")
        retrieve_mock.assert_called_once_with(
            vectordb=vectordb,
            user_query="반납 절차 알려줘",
            refined_query="물품 반납 절차",
            top_k=chain.RETRIEVER_TOP_K,
            search_mode=chain.RAG_SEARCH_MODE,
            use_hybrid=chain.USE_HYBRID_RETRIEVAL,
        )
        self.assertEqual(result["answer"], "반납 절차 답변")
        self.assertEqual(result["diagnostics"]["classification"], "NEED_RAG")
        self.assertEqual(result["diagnostics"]["tool_rag_policy"], "rag_only")
        self.assertEqual(result["diagnostics"]["final_context_count"], 2)

    def test_greeting_routes_to_no_rag_and_skips_retrieval(self):
        self.base_llm.invoke.return_value = AIMessage(content="안녕하세요.")

        with (
            patch.object(chain, "classify_question", return_value="NO_RAG") as classify_mock,
            patch.object(chain, "refine_query") as refine_mock,
            patch.object(chain, "retrieve_candidate_docs") as retrieve_mock,
        ):
            result = run_rag_chain(self.base_llm, MagicMock(), "안녕")

        classify_mock.assert_called_once_with(self.base_llm, "안녕")
        refine_mock.assert_not_called()
        retrieve_mock.assert_not_called()
        self.assertEqual(result["answer"], "안녕하세요.")
        self.assertEqual(result["attribution"], [])
        self.assertEqual(result["diagnostics"]["classification"], "NO_RAG")
        self.assertEqual(result["diagnostics"]["tool_rag_policy"], "no_rag")

    def test_no_context_response_when_threshold_filter_removes_candidates(self):
        docs = [(_doc("far-doc"), 9.9)]

        with (
            patch.object(chain, "classify_question", return_value="NEED_RAG"),
            patch.object(chain, "refine_query", return_value="물품 불용 절차"),
            patch.object(chain, "retrieve_candidate_docs", return_value=docs),
            patch.object(chain, "filter_retrieved_docs", return_value=[]) as filter_mock,
        ):
            result = run_rag_chain(self.base_llm, MagicMock(), "불용 절차 알려줘")

        filter_mock.assert_called_once_with(docs, strategy=chain.RAG_THRESHOLD_STRATEGY)
        self.assertEqual(result["answer"], chain.NO_CONTEXT_RESPONSE)
        self.assertEqual(result["diagnostics"]["retrieved_count"], 1)
        self.assertEqual(result["diagnostics"]["filtered_count"], 0)
        self.assertEqual(result["diagnostics"]["final_context_count"], 0)

    def test_reranker_orders_final_context_when_enabled(self):
        vectordb = MagicMock(name="vectordb")
        docs = [
            (_doc("vector-1", category="물품 반납 관리", chapter="반납"), 0.10),
            (_doc("vector-2", category="물품 불용 관리", chapter="불용"), 0.11),
            (_doc("vector-3", category="물품 취득 관리", chapter="취득"), 0.12),
        ]
        reranked = [
            (docs[2][0], docs[2][1], 0.98),
            (docs[0][0], docs[0][1], 0.77),
            (docs[1][0], docs[1][1], 0.66),
        ]
        reranker = MagicMock(name="reranker")
        reranker.rerank_with_scores.return_value = reranked
        self.base_llm.invoke.return_value = AIMessage(content="reranked answer")

        with (
            patch.object(chain, "USE_RERANKING", True),
            patch.object(chain, "RERANK_TOP_N", 4),
            patch.object(chain, "RERANK_CANDIDATE_K", 10),
            patch.object(chain, "classify_question", return_value="NEED_RAG"),
            patch.object(chain, "refine_query", return_value="물품 반납 불용 차이"),
            patch.object(chain, "retrieve_candidate_docs", return_value=docs),
            patch.object(chain, "filter_retrieved_docs", return_value=docs),
            patch.object(chain, "CrossEncoderReranker", return_value=reranker) as reranker_cls,
        ):
            result = run_rag_chain(self.base_llm, vectordb, "반납과 불용 차이 알려줘")

        reranker_cls.assert_called_once_with(chain.RERANKER_MODEL_NAME)
        reranker.rerank_with_scores.assert_called_once_with(
            query="물품 반납 불용 차이",
            docs_with_scores=docs,
            top_n=4,
        )
        self.assertEqual(
            [entry["doc_id"] for entry in result["diagnostics"]["final_context_scores"]],
            ["vector-3", "vector-1", "vector-2"],
        )
        self.assertEqual(
            [item["doc_id"] for item in result["attribution"]],
            ["vector-3", "vector-1", "vector-2"],
        )
        self.assertIn("rerank_score=0.980000", result["diagnostics"]["final_context_text"])

    def test_tool_lookup_question_skips_rag(self):
        fake_tool = MagicMock(name="get_item_detail_info")
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
            patch.object(chain, "retrieve_candidate_docs") as retrieve_mock,
        ):
            result = run_rag_chain(self.base_llm, MagicMock(), "노트북 재고 조회해줘")

        fake_tool.invoke.assert_called_once_with({"asset_name": "노트북"})
        classify_mock.assert_not_called()
        retrieve_mock.assert_not_called()
        self.assertEqual(result["answer"], "노트북 재고는 10대입니다.")
        self.assertEqual(result["attribution"], [])
        self.assertEqual(result["diagnostics"]["tool_rag_policy"], "tool_only")
        self.assertEqual(result["diagnostics"]["tool_result_statuses"], ["success"])

    def test_tool_and_procedure_question_uses_tool_result_and_rag_context(self):
        vectordb = MagicMock(name="vectordb")
        fake_tool = MagicMock(name="get_item_detail_info")
        fake_tool.invoke.return_value = json.dumps(
            {"results": [{"g2b_name": "노트북", "status": "운용"}]},
            ensure_ascii=False,
        )
        docs = [
            (
                _doc(
                    "manual-1",
                    content="불용 기준은 사용 불가 물품을 대상으로 합니다.",
                    category="물품 불용 관리",
                    chapter="불용",
                ),
                0.12,
            ),
            (
                _doc(
                    "manual-2",
                    content="불용 절차는 신청 후 확정 흐름을 따릅니다.",
                    category="물품 불용 관리",
                    chapter="불용",
                ),
                0.14,
            ),
        ]
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
        self.base_llm.invoke.return_value = AIMessage(
            content="노트북은 운용 중이며, 불용 절차는 다음과 같습니다."
        )

        with (
            patch.object(chain, "USE_RERANKING", False),
            patch.object(chain, "TOOL_MAP", {"get_item_detail_info": fake_tool}),
            patch.object(chain, "classify_question") as classify_mock,
            patch.object(chain, "refine_query", return_value="노트북 불용 절차"),
            patch.object(chain, "retrieve_candidate_docs", return_value=docs) as retrieve_mock,
            patch.object(chain, "filter_retrieved_docs", return_value=docs),
        ):
            result = run_rag_chain(self.base_llm, vectordb, "노트북 상태 조회하고 불용 절차도 알려줘")

        classify_mock.assert_not_called()
        retrieve_mock.assert_called_once()
        self.assertEqual(result["answer"], "노트북은 운용 중이며, 불용 절차는 다음과 같습니다.")
        self.assertEqual(result["diagnostics"]["classification"], "NEED_RAG")
        self.assertEqual(result["diagnostics"]["tool_rag_policy"], "mixed_with_rag_context")
        self.assertEqual(result["diagnostics"]["tool_result_statuses"], ["success"])
        final_prompt = self.base_llm.invoke.call_args[0][0][0].content
        self.assertIn("[도구 조회 결과]", final_prompt)
        self.assertIn("[참고 자료]", final_prompt)
        self.assertIn("불용 기준은 사용 불가 물품을 대상으로 합니다.", final_prompt)


class TestRoutingHelpers(unittest.TestCase):
    def test_retrieve_candidate_docs_uses_role_specific_retrieve_docs(self):
        vectordb = MagicMock(name="vectordb")
        qa_doc = _doc("qa-1", doc_type="qa")
        manual_doc = _doc("manual-1", doc_type="manual_chunk")
        faq_doc = _doc("faq-1", doc_type="faq", source="faq")

        def fake_retrieve_docs(**kwargs):
            doc_type = kwargs["metadata_filter"]["doc_type"]
            if doc_type == "qa":
                return [(qa_doc, 0.20)]
            if doc_type == "manual_chunk":
                return [(manual_doc, 0.10)]
            return [(faq_doc, 0.15)]

        with patch.object(chain, "retrieve_docs", side_effect=fake_retrieve_docs) as retrieve_docs_mock:
            results = chain.retrieve_candidate_docs(
                vectordb=vectordb,
                user_query="원 질문",
                refined_query="정제 질문",
                top_k=7,
                search_mode="refined",
            )

        retrieve_docs_mock.assert_has_calls(
            [
                call(
                    vectordb=vectordb,
                    query="정제 질문",
                    top_k=7,
                    metadata_filter={"doc_type": "qa"},
                    use_hybrid=chain.USE_HYBRID_RETRIEVAL,
                ),
                call(
                    vectordb=vectordb,
                    query="정제 질문",
                    top_k=7,
                    metadata_filter={"doc_type": "manual_chunk"},
                    use_hybrid=chain.USE_HYBRID_RETRIEVAL,
                ),
                call(
                    vectordb=vectordb,
                    query="정제 질문",
                    top_k=chain.FAQ_RETRIEVAL_LIMIT,
                    metadata_filter={"doc_type": "faq"},
                    use_hybrid=chain.USE_HYBRID_RETRIEVAL,
                ),
            ]
        )
        self.assertEqual([doc.metadata["doc_id"] for doc, _score in results], ["manual-1", "faq-1", "qa-1"])

    def test_classify_question_parses_allowed_labels_and_defaults_to_need_rag(self):
        fake_chain = MagicMock()

        with patch.object(chain, "_build_text_chain", return_value=fake_chain):
            fake_chain.invoke.return_value = "NO_RAG"
            self.assertEqual(chain.classify_question(MagicMock(), "안녕"), "NO_RAG")

            fake_chain.invoke.return_value = "NEED_RAG"
            self.assertEqual(chain.classify_question(MagicMock(), "반납 절차 알려줘"), "NEED_RAG")

            fake_chain.invoke.return_value = '{"classification": "NO_RAG"}'
            self.assertEqual(chain.classify_question(MagicMock(), "고마워"), "NO_RAG")

            fake_chain.invoke.return_value = "판단: NO_RAG"
            self.assertEqual(chain.classify_question(MagicMock(), "안녕"), "NEED_RAG")

            fake_chain.invoke.return_value = "잘 모르겠습니다"
            self.assertEqual(chain.classify_question(MagicMock(), "불용 기준 알려줘"), "NEED_RAG")

    def test_refine_query_returns_first_nonempty_line_and_falls_back(self):
        fake_chain = MagicMock()

        with patch.object(chain, "_build_text_chain", return_value=fake_chain):
            fake_chain.invoke.return_value = "```\n물품 반납 절차\n추가 설명\n```"
            self.assertEqual(chain.refine_query(MagicMock(), "반납 어떻게 해?"), "물품 반납 절차")

            fake_chain.invoke.return_value = "물품 조회 방법"
            self.assertEqual(
                chain.refine_query(MagicMock(), "자산번호 A-12345 반납 절차 알려줘"),
                "물품 조회 방법 반납 자산번호 A-12345",
            )

            fake_chain.invoke.return_value = ""
            self.assertEqual(chain.refine_query(MagicMock(), "원 질문"), "원 질문")

            fake_chain.invoke.side_effect = RuntimeError("llm down")
            self.assertEqual(chain.refine_query(MagicMock(), "원 질문"), "원 질문")

    def test_context_sort_prefers_manual_and_faq_over_qa(self):
        docs = [
            _doc("qa-1", doc_type="qa"),
            _doc("manual-1", doc_type="manual_chunk"),
            _doc("faq-1", doc_type="faq", source="faq"),
        ]

        sorted_docs = chain._sort_docs_for_context(docs)

        self.assertEqual([doc.metadata["doc_id"] for doc in sorted_docs], ["manual-1", "faq-1", "qa-1"])

    def test_filter_retrieved_docs_abstains_when_minimum_evidence_is_missing(self):
        docs = [(_doc("only-doc"), 0.10)]

        self.assertEqual(chain.filter_retrieved_docs(docs, threshold=0.95, strategy="fixed"), [])


if __name__ == "__main__":
    unittest.main()
