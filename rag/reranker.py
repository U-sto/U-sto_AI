from sentence_transformers import CrossEncoder
from app.config import RERANK_DEBUG, USE_RERANKING

_reranker_instance = None

def get_reranker(model_name: str):
    global _reranker_instance
    if _reranker_instance is None:
        if RERANK_DEBUG:
            print(f"[RERANKER] 모델 로딩: {model_name}")
        _reranker_instance = CrossEncoder(model_name)
    return _reranker_instance

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        if RERANK_DEBUG:
            print(f"[RERANKER] 모델 로딩: {model_name}")

        self.model = get_reranker(model_name)

        if RERANK_DEBUG:
            print("[RERANKER] 모델 로딩 완료")

    def rerank(self, query: str, docs: list, top_n: int):
        if RERANK_DEBUG:
            print(f"[RERANKER] Re-ranking 시작 | 후보 수: {len(docs)}")

        pairs = [(query, doc.page_content) for doc, _ in docs]
        scores = self.model.predict(pairs)

        # 디버깅: 상위 5개 score 출력
        if RERANK_DEBUG: 
            for i, score in enumerate(scores[:5]):
                print(f"[RERANKER] score[{i}] = {score}")

        reranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        if RERANK_DEBUG:
            print("[RERANKER] Re-ranking 완료")
        return [doc for (doc, _), _ in reranked[:top_n]]
