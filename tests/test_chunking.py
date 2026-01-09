import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.loader import split_text

class TestTextSplitting(unittest.TestCase):

    def test_short_text(self):
        """1. 텍스트가 청크 크기보다 짧을 때 테스트"""
        text = "짧은 문장입니다."
        chunks = split_text(text, chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "짧은 문장입니다.")

    def test_exact_split_no_overlap_issue(self):
        """2. 공백이 없는 긴 텍스트 (Look-back 실패 시 강제 절단 테스트)"""
        text = "A" * 100  # 띄어쓰기 없는 100글자
        chunks = split_text(text, chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 2) # 50, 50으로 나뉨
        self.assertEqual(chunks[0], "A" * 50)
        self.assertEqual(chunks[1], "A" * 50) # start가 next_step으로 이동했다가 겹쳐서 조정됨

    def test_overlap_logic(self):
        """3. 오버랩(겹치는 구간)이 잘 작동하는지 테스트"""
        text = "1234567890"
        # 크기 5, 겹침 2 -> [12345], [45678], [7890] 예상을 검증
        # 로직상 look-back이 없으므로 단순 잘림
        chunks = split_text(text, chunk_size=5, overlap=2)
        
        # 첫 번째 청크 끝부분과 두 번째 청크 앞부분이 겹치는지 확인
        self.assertTrue(chunks[0].endswith(chunks[1][:2]))

    def test_infinite_loop_prevention(self):
        """4. chunk_size <= overlap 일 때 에러 발생(무한루프 방지) 테스트"""
        with self.assertRaises(ValueError):
            split_text("아무 텍스트", chunk_size=50, overlap=50)
        
        with self.assertRaises(ValueError):
            split_text("아무 텍스트", chunk_size=50, overlap=60)

    def test_natural_split(self):
        """5. 자연스러운 문장 분리 (공백/마침표 기준) 테스트"""
        # 10글자에서 자르려는데, 9번째에 마침표가 있음
        text = "Hello World. This is a test."
        # chunk_size=13, overlap=0
        # "Hello World." (12자) -> 여기서 잘려야 함
        chunks = split_text(text, chunk_size=13, overlap=5)
        
        self.assertIn("Hello World.", chunks[0])
        self.assertIn("This is", chunks[1])

if __name__ == '__main__':
    unittest.main()