from sentence_transformers import CrossEncoder
from app.config import RERANK_DEBUG

_reranker_instances = {}

def get_reranker(model_name: str):
    if model_name not in _reranker_instances:
        try:
            if RERANK_DEBUG:
                print(f"[RERANKER] 모델 로딩: {model_name}")
            _reranker_instances[model_name] = CrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(f"CrossEncoder 모델 로딩 실패: {e}")
    return _reranker_instances[model_name]

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = get_reranker(model_name)

        if RERANK_DEBUG:
            print("[RERANKER] 모델 로딩 완료")

    def rerank(self, query: str, docs_with_scores: list, top_n: int):
        """
        Re-rank candidate documents for a query using a Cross-Encoder.

        Parameters
        ----------
        query : str
            사용자 질의 문자열
        docs : list
            (Document, retrieval_score) 형태의 리스트
        top_n : int
            반환할 상위 문서 개수

        Returns
        -------
        list
            Cross-Encoder 점수 기준으로 재정렬된 Document 리스트
        """
        try:
            pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
            scores = self.model.predict(pairs)
        except Exception as e:
            if RERANK_DEBUG:
                print(f"[RERANKER] 예외 발생, re-ranking 생략: {e}")
            # fallback: retrieval 순서 유지
            return [doc for doc, _ in docs_with_scores[:top_n]]

        # 디버깅: 상위 5개 score 출력
        if RERANK_DEBUG: 
            for i, score in enumerate(scores[:5]):
                print(f"[RERANKER] score[{i}] = {score}")

        reranked = sorted(
            zip(docs_with_scores, scores),
            key=lambda x: x[1],
            reverse=True
        )
        if RERANK_DEBUG:
            print("[RERANKER] Re-ranking 완료")
        return [doc for (doc, _), _ in reranked[:top_n]]
