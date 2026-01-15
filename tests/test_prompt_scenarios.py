# S10 시나리오는
# LLM의 실제 응답을 검증하는 보안 테스트가 아니라
# 프롬프트에 '안전 지침'이 항상 포함되는지 확인하는 정책 테스트임

import json  # 시나리오 파일 로드용 모듈 import
import os  # 파일 경로 조립용 모듈 import
import unittest  # unittest 프레임워크 import

import app.config as config  # config 모듈 자체 참조 import
from rag.prompt import assemble_prompt  # 프롬프트 조립 함수 import


class TestPromptScenarios(unittest.TestCase):  # 시나리오 기반 프롬프트 테스트 클래스 정의
    """
    프롬프트 시나리오(JSON) 기반으로
    assemble_prompt 함수의 정책 일관성을 검증하는 테스트 클래스
    """
    @classmethod
    def setUpClass(cls):
        """프롬프트 시나리오 JSON 로드"""
        base_dir = os.path.dirname(__file__)  
        scenario_path = os.path.join(base_dir, "prompt_scenarios.json")
        with open(scenario_path, "r", encoding="utf-8") as f:
            cls.scenarios = json.load(f)  

    def setUp(self):
        """테스트 실행 전 config 플래그 백업 및 초기화"""
        self._orig_system = config.ENABLE_SYSTEM_PROMPT
        self._orig_safety = config.ENABLE_SAFETY_PROMPT
        self._orig_func = config.ENABLE_FUNCTION_DECISION_PROMPT
        config.ENABLE_SYSTEM_PROMPT = True
        config.ENABLE_SAFETY_PROMPT = True
        config.ENABLE_FUNCTION_DECISION_PROMPT = True

    def tearDown(self):
        """테스트 종료 후 config 플래그 원복"""
        config.ENABLE_SYSTEM_PROMPT = self._orig_system
        config.ENABLE_SAFETY_PROMPT = self._orig_safety
        config.ENABLE_FUNCTION_DECISION_PROMPT = self._orig_func

    def test_all_scenarios(self):
        """모든 프롬프트 시나리오를 순회하며 정책 검증"""
        for sc in self.scenarios:
            with self.subTest(scenario_id=sc["id"]):
                prompt = assemble_prompt(
                    context=sc.get("context", ""),
                    question=sc.get("question", "")
                )

                self.assertIsInstance(prompt, str)
                self.assertGreater(len(prompt), 0)

                for token in sc.get("must_include", []):
                    self.assertIn(
                        token,
                        prompt,
                        f"[{sc['id']}] expected token not found: {token}",
                    )

                for token in sc.get("must_not_include", []):
                    self.assertNotIn(
                        token,
                        prompt,
                        f"[{sc['id']}] forbidden token found: {token}",
    )



if __name__ == "__main__":  # 직접 실행 시 엔트리포인트 조건
    unittest.main()  # unittest 러너 실행
