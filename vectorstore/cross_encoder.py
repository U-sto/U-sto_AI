# cross_encoder.py

from sentence_transformers import CrossEncoder  # Cross-Encoder 로드

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        # Cross-Encoder 모델 초기화
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs_with_scores: list, top_n: int):
        # (query, document) 쌍 생성
        pairs = [
            (query, doc.page_content)  # 질의 + 문서 내용
            for doc, _ in docs_with_scores
        ]

        # Cross-Encoder 점수 예측
        scores = self.model.predict(pairs)

        # 문서 + 점수 결합
        reranked = [
            (docs_with_scores[i][0], scores[i])  # (doc, cross_score)
            for i in range(len(scores))
        ]

        # 점수 내림차순 정렬
        reranked.sort(key=lambda x: x[1], reverse=True)

        # 상위 top_n개 반환
        return reranked[:top_n]
