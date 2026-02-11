import unittest
from unittest.mock import patch
import app.config as config
from rag.prompt import assemble_prompt
from rag.faq_service import get_relevant_faq_string

_MISSING = object()
class TestFAQPromptLogic(unittest.TestCase):
    
    def setUp(self):
        # 백업: 전역 설정 값 저장 (테스트 간 누수 방지)
        self._orig_enable_faq_prompt = getattr(config, "ENABLE_FAQ_PROMPT", _MISSING)
        # 만약 설정 값이 존재하지 않았다면, 테스트용 기본값을 설정
        if self._orig_enable_faq_prompt is _MISSING:
            config.ENABLE_FAQ_PROMPT = False
        self.context = "기존 검색 컨텍스트"
        self.question = "반납이랑 불용 차이가 뭐야?" # FAQ 키워드 '불용차이' 포함 가정
    
    def tearDown(self):
        # 복원: 전역 설정 값을 원래대로 되돌림
        if self._orig_enable_faq_prompt is _MISSING:
            # 원래 존재하지 않았던 경우에는 속성을 제거
            delattr(config, "ENABLE_FAQ_PROMPT")
        else:
            config.ENABLE_FAQ_PROMPT = self._orig_enable_faq_prompt


    @patch('rag.prompt.get_relevant_faq_string')
    def test_faq_flag_enabled_with_match(self, mock_get_faq):
        """
        Case 1: ENABLE_FAQ_PROMPT = True이고, 관련 FAQ가 있을 때
        -> 프롬프트에 FAQ 섹션이 포함되어야 한다.
        """
        # 설정: 플래그 ON
        config.ENABLE_FAQ_PROMPT = True
        # 설정: 서비스가 유효한 텍스트를 반환한다고 가정
        mock_get_faq.return_value = "Q: 반납... A: 불용..."

        result = assemble_prompt(self.context, self.question)

        self.assertIn("[FAQ 지식 베이스 (관련 내용)]", result)
        self.assertIn("Q: 반납... A: 불용...", result)
        # assemble_prompt가 내부적으로 build_faq_prompt 호출 시 질문을 잘 넘겼는지 확인
        mock_get_faq.assert_called_with(self.question)


    @patch('rag.prompt.get_relevant_faq_string')
    def test_faq_flag_enabled_no_match(self, mock_get_faq):
        """
        Case 2: ENABLE_FAQ_PROMPT = True이지만, 관련 FAQ가 없을 때
        -> 프롬프트에 FAQ 헤더가 없어야 한다. (토큰 절약 확인)
        """
        config.ENABLE_FAQ_PROMPT = True
        # 설정: 서비스가 빈 문자열 반환 (매칭 실패)
        mock_get_faq.return_value = ""

        result = assemble_prompt(self.context, "이상한 질문")

        self.assertNotIn("[FAQ 지식 베이스 (관련 내용)]", result)


    @patch('rag.prompt.get_relevant_faq_string')
    def test_faq_flag_disabled(self, mock_get_faq):
        """
        Case 3: ENABLE_FAQ_PROMPT = False일 때
        -> 관련 FAQ가 있어도 프롬프트에 포함되면 안 된다.
        """
        # 설정: 플래그 OFF
        config.ENABLE_FAQ_PROMPT = False
        # 설정: 서비스는 데이터를 주려고 하지만...
        mock_get_faq.return_value = "Q: 데이터 A: 있음"

        result = assemble_prompt(self.context, self.question)

        # 결과: 아예 포함되지 않아야 함
        self.assertNotIn("[FAQ 지식 베이스 (관련 내용)]", result)
        self.assertNotIn("Q: 데이터 A: 있음", result)


    @patch('rag.faq_service._ensure_faq_loaded', return_value=None)
    def test_service_keyword_filtering_real_logic(self, mock_ensure_faq_loaded):
        """
        Case 4: (서비스 로직 검증) 실제 키워드 매칭이 동작하는지 확인
        """
        from rag.faq_service import get_relevant_faq_string
        # 각 테스트 실행마다 새로운 FAQ 캐시 데이터를 주입하여
        # 프로덕션 코드에서의 변이가 테스트 간에 누수되지 않도록 한다.
        with patch('rag.faq_service._FAQ_CACHE_DATA', [
            {"question": "Q1", "answer": "A1", "keywords": ["테스트"]}
        ]):
            # 1. 키워드 매칭 성공
            res_match = get_relevant_faq_string("이건 테스트 질문이야")
            self.assertIn("Q: Q1", res_match)
            # 2. 키워드 매칭 실패
            res_no_match = get_relevant_faq_string("이건 엉뚱한 질문이야")
            self.assertEqual(res_no_match, "")
            # 3. 목록 요청 (사용자님의 최신 로직: 질문+답변 쌍 전체 출력)
            res_list = get_relevant_faq_string("FAQ 보여줘")
            # 헤더 문구 확인 (내용 목록으로 수정됨)
            self.assertIn("[FAQ 전체 내용 목록]", res_list)
            # 질문과 답변이 쌍으로 잘 들어있는지 확인
            self.assertIn("Q: Q1", res_list)
            self.assertIn("A: A1", res_list)
    @patch('rag.faq_service._ensure_faq_loaded') # [추가] 실제 로드 로직 실행 방지 (No-op)
    def test_normalization_matching(self, mock_ensure_loaded):
        """
        정규화(Normalization) 동작 검증:
        공백, 특수문자, 대소문자가 달라도 키워드를 찾아내야 한다.
        """
        # 각 테스트 실행마다 새로운 FAQ 캐시 데이터를 주입하여
        # 프로덕션 코드에서의 변이가 테스트 간에 누수되지 않도록 한다.
        with patch('rag.faq_service._FAQ_CACHE_DATA', [
            {"question": "Q_불용", "answer": "A_불용", "keywords": ["불용차이", "반납불용"]},
            {"question": "Q_G2B", "answer": "A_G2B", "keywords": ["G2B번호"]}
        ]):
            # mock_ensure_loaded는 아무 동작도 하지 않으므로, 
            # 위에서 주입한 _FAQ_CACHE_DATA가 그대로 유지됩니다.
            # Case 1: 공백이 들어간 경우 ("불용 차이" -> "불용차이")
            res_space = get_relevant_faq_string("반납이랑 불용 차이가 궁금해요")
            self.assertIn("Q_불용", res_space)
            # Case 2: 특수문자가 섞인 경우 ("불용-차이??")
            res_symbol = get_relevant_faq_string("반납/불용-차이?? 알려줘")
            self.assertIn("Q_불용", res_symbol)
            # Case 3: 대소문자 및 공백 혼합 ("g2b 번호")
            res_case = get_relevant_faq_string("g2b 번호가 뭐야?")
            self.assertIn("Q_G2B", res_case)
            # Case 4: 매칭되지 않아야 하는 경우
            res_fail = get_relevant_faq_string("완전 다른 질문")
            self.assertEqual(res_fail, "")

if __name__ == '__main__':
    unittest.main()