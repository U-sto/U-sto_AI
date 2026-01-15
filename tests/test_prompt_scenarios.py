import json  # 시나리오 파일 로드용 모듈 import
import os  # 파일 경로 조립용 모듈 import
import unittest  # unittest 프레임워크 import

import app.config as config  # config 모듈 자체 참조 import
from rag.prompt import assemble_prompt  # 프롬프트 조립 함수 import


class TestPromptScenarios(unittest.TestCase):  # 시나리오 기반 프롬프트 테스트 클래스 정의
    @classmethod
    def setUpClass(cls):  # 클래스 단위로 1회만 실행되는 셋업 정의
        base_dir = os.path.dirname(__file__)  # 현재 테스트 파일 디렉터리 경로 계산
        scenario_path = os.path.join(base_dir, "prompt_scenarios.json")  # 시나리오 JSON 경로 조립
        with open(scenario_path, "r", encoding="utf-8") as f:  # JSON 파일을 UTF-8로 오픈
            cls.scenarios = json.load(f)  # JSON 내용을 파이썬 객체로 로드

    def setUp(self):  # 각 테스트 실행 전 호출되는 셋업 정의
        self._orig_system = config.ENABLE_SYSTEM_PROMPT  # 원래 시스템 플래그 백업
        self._orig_safety = config.ENABLE_SAFETY_PROMPT  # 원래 세이프티 플래그 백업
        self._orig_func = config.ENABLE_FUNCTION_DECISION_PROMPT  # 원래 함수판단 플래그 백업
        config.ENABLE_SYSTEM_PROMPT = True  # 테스트 기본값으로 시스템 프롬프트 활성화
        config.ENABLE_SAFETY_PROMPT = True  # 테스트 기본값으로 안전지침 프롬프트 활성화
        config.ENABLE_FUNCTION_DECISION_PROMPT = True  # 테스트 기본값으로 함수판단 프롬프트 활성화

    def tearDown(self):  # 각 테스트 실행 후 호출되는 정리 함수 정의
        config.ENABLE_SYSTEM_PROMPT = self._orig_system  # 시스템 플래그 원복
        config.ENABLE_SAFETY_PROMPT = self._orig_safety  # 세이프티 플래그 원복
        config.ENABLE_FUNCTION_DECISION_PROMPT = self._orig_func  # 함수판단 플래그 원복

    def test_all_scenarios(self):  # 10개 시나리오 전체를 순회 검증하는 테스트 정의
        for sc in self.scenarios:  # 시나리오 리스트 순회
            with self.subTest(scenario_id=sc["id"]):  # 실패 시 어떤 케이스인지 식별 가능한 서브테스트
                prompt = assemble_prompt(  # 프롬프트 조립 실행
                    context=sc.get("context", ""),  # 시나리오 컨텍스트 주입
                    question=sc.get("question", "")  # 시나리오 질문 주입
                )

                self.assertIsInstance(prompt, str)  # 결과가 문자열 타입인지 확인
                self.assertGreater(len(prompt), 0)  # 결과가 빈 문자열이 아닌지 확인

                for token in sc.get("must_include", []):  # 반드시 포함되어야 할 토큰 리스트 순회
                    self.assertIn(token, prompt)  # 포함 여부 단언

                for token in sc.get("must_not_include", []):  # 포함되면 안 되는 토큰 리스트 순회
                    self.assertNotIn(token, prompt)  # 미포함 여부 단언


if __name__ == "__main__":  # 직접 실행 시 엔트리포인트 조건
    unittest.main()  # unittest 러너 실행
