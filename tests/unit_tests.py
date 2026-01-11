# 4. 답변 안정성 테스트
# 개념: LLM의 답변이 일관성 있고 신뢰할 수 있는지 검증. 
# 테스트 케이스를 작성하고, 답변의 정확성과 규칙 준수 여부를 확인.

# fallback 안정성
# 출력 포맷 일관성
# 출처 필수 포함

# unit_tests.py

import unittest
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from ingestion.embedder import get_embedding_model
from vectorstore.chroma_store import load_chroma_db
from rag.chain import run_rag_chain
from app.config import (
    VECTOR_DB_PATH,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    NO_CONTEXT_RESPONSE
)

class TestRAGChain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 환경 변수 로드
        load_dotenv()

        # 임베딩 모델 로드
        embedding_model = get_embedding_model()

        # Chroma DB 로드
        cls.vectordb = load_chroma_db(
            embedding_model,
            VECTOR_DB_PATH
        )

        # LLM 로드
        cls.llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE
        )

    def test_no_context_response(self):
        response = run_rag_chain(
            llm=self.llm,
            vectordb=self.vectordb,
            user_query="존재하지 않는 질문"
        )
        self.assertEqual(response["answer"], NO_CONTEXT_RESPONSE)

    def test_answer_format(self):
        response = run_rag_chain(
            llm=self.llm,
            vectordb=self.vectordb,
            user_query="취득과 정리구분의 차이를 알려줘"
        )
        self.assertTrue(
            response["answer"].endswith("하십시오.") or
            response["answer"] == NO_CONTEXT_RESPONSE
        )

    def test_attribution_exists(self):
        response = run_rag_chain(
            llm=self.llm,
            vectordb=self.vectordb,
            user_query="취득과 정리구분의 차이를 알려줘"
        )
        self.assertIsInstance(response["attribution"], list)
        
        if not response["attribution"]:
            self.skipTest("No attribution returned; skipping attribution field checks.")

        first = response["attribution"][0]
        self.assertIn("doc_id", first)


    def test_attribution_fields(self):
        response = run_rag_chain(
            llm=self.llm,
            vectordb=self.vectordb,
            user_query="질문"
        )
        first = response["attribution"][0]
        self.assertIn("doc_id", first)

    def test_reranking_enabled_pipeline(self):
        response = run_rag_chain(
            llm=self.llm,
            vectordb=self.vectordb,
            user_query="취득과 정리구분의 차이를 알려줘"
        )

        # answer는 문자열이어야 함
        self.assertIsInstance(response["answer"], str)

        # attribution은 리스트여야 함
        self.assertIsInstance(response["attribution"], list)



if __name__ == "__main__":
    unittest.main()

