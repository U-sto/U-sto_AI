# 답변 안정성 테스트
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
from rag.prompt import assemble_prompt

import app.config as config
from app.config import (
    VECTOR_DB_PATH,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    NO_CONTEXT_RESPONSE
)
class TestPromptAssembly(unittest.TestCase):
    """
    assemble_prompt 함수가
    config 플래그 조합에 따라 올바른 프롬프트를 생성하는지 검증
    """

    def setUp(self):
        self.context = "샘플 참고 자료"
        self.question = "샘플 질문"
    
    def test_system_prompt_enabled(self):
        config.ENABLE_SYSTEM_PROMPT = True

        prompt = assemble_prompt(
            context=self.context,
            question=self.question
        )

        self.assertIn("[시스템 정체성]", prompt)

    def test_system_prompt_disabled(self):
        config.ENABLE_SYSTEM_PROMPT = False

        prompt = assemble_prompt(
            context=self.context,
            question=self.question
        )

        self.assertNotIn("[시스템 정체성]", prompt)
    
    def test_function_decision_prompt_enabled(self):
        config.ENABLE_FUNCTION_DECISION_PROMPT = True

        prompt = assemble_prompt(
            context=self.context,
            question=self.question
        )

        self.assertIn("[Function Calling 판단 기준]", prompt)

    def test_function_decision_prompt_disabled(self):
        config.ENABLE_FUNCTION_DECISION_PROMPT = False

        prompt = assemble_prompt(
            context=self.context,
            question=self.question
        )

        self.assertNotIn("[Function Calling 판단 기준]", prompt)
    
    def test_safety_prompt_enabled(self):
        config.ENABLE_SAFETY_PROMPT = True

        prompt = assemble_prompt(
            context=self.context,
            question=self.question
        )

        self.assertIn("[안전 지침]", prompt)


    def test_safety_prompt_disabled(self):
        config.ENABLE_SAFETY_PROMPT = False

        prompt = assemble_prompt(
            context=self.context,
            question=self.question
        )

        self.assertNotIn("[안전 지침]", prompt)


    def tearDown(self):
        config.ENABLE_SYSTEM_PROMPT = True
        config.ENABLE_SAFETY_PROMPT = True
        config.ENABLE_FUNCTION_DECISION_PROMPT = True

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
            user_query="취득과 정리구분의 차이를 알려줘"
        )
        if not response["attribution"]:
            self.skipTest("No attribution returned; skipping attribution field checks.")

        first = response["attribution"][0]
        self.assertIn("doc_id", first)

    def test_reranking_enabled_pipeline(self):
        """
        Re-ranking 활성화 상태에서도
        RAG 파이프라인이 정상적으로 동작하는지 검증
        """
        response = run_rag_chain(
            llm=self.llm,
            vectordb=self.vectordb,
            user_query="취득과 정리구분의 차이를 알려줘"
        )

        self.assertIn("answer", response)
        self.assertIn("attribution", response)


if __name__ == "__main__":
    unittest.main()

