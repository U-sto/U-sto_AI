import sys
import logging
from pathlib import Path

try:
    from app.config import RERANK_DEBUG
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.config import RERANK_DEBUG

_reranker_instances = {}
logger = logging.getLogger(__name__)

def get_reranker(model_name: str):
    if model_name not in _reranker_instances:
        try:
            from sentence_transformers import CrossEncoder

            if RERANK_DEBUG:
                print(f"[RERANKER] 모델 로딩: {model_name}")
            _reranker_instances[model_name] = CrossEncoder(model_name)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "sentence-transformers가 설치되어 있지 않아 reranker를 사용할 수 없습니다."
            ) from e
        except Exception as e:
            raise RuntimeError(f"CrossEncoder 모델 로딩 실패: {e}")
    return _reranker_instances[model_name]

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = get_reranker(model_name)

        if RERANK_DEBUG:
            print("[RERANKER] 모델 로딩 완료")

    def rerank_with_scores(self, query: str, docs_with_scores: list, top_n: int) -> list[tuple]:
        """
        Re-rank candidate documents and keep both retrieval and rerank scores.

        Parameters
        ----------
        query : str
            사용자 질의 문자열
        docs_with_scores : list
            (Document, retrieval_score) 형태의 리스트
        top_n : int
            반환할 상위 문서 개수

        Returns
        -------
        list
            (Document, retrieval_score, rerank_score) 형태의 리스트
        """
        if not docs_with_scores or top_n <= 0:
            return []

        try:
            pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
            raw_scores = self.model.predict(pairs)
            scores = [float(score) for score in raw_scores]
        except Exception as e:
            if RERANK_DEBUG:
                logger.warning("[RERANKER] 예외 발생, re-ranking 생략: %s", e)
            # fallback: retrieval 순서 유지
            return [
                (doc, retrieval_score, None)
                for doc, retrieval_score in docs_with_scores[:top_n]
            ]

        # 디버깅: 상위 5개 score 출력
        if RERANK_DEBUG: 
            for i, score in enumerate(scores[:5]):
                logger.debug("[RERANKER] score[%d] = %.6f", i, score)

        reranked = sorted(
            zip(docs_with_scores, scores),
            key=lambda x: x[1],
            reverse=True
        )
        if RERANK_DEBUG:
            logger.debug("[RERANKER] Re-ranking 완료")

        return [
            (doc, retrieval_score, rerank_score)
            for (doc, retrieval_score), rerank_score in reranked[:top_n]
        ]

    def rerank(self, query: str, docs_with_scores: list, top_n: int):
        """
        Re-rank candidate documents for a query using a Cross-Encoder.

        Returns
        -------
        list
            Cross-Encoder 점수 기준으로 재정렬된 Document 리스트
        """
        reranked = self.rerank_with_scores(query, docs_with_scores, top_n)
        return [doc for doc, _retrieval_score, _rerank_score in reranked]
