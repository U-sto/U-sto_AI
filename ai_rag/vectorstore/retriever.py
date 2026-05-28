import logging
from typing import List, Tuple


logger = logging.getLogger(__name__)


def retrieve_docs(vectordb, query: str, top_k: int) -> List[Tuple]:
    """
    Chroma VectorStore를 통해 유사 문서 검색을 수행한다.

    반환값은 [(Document, score), ...] 형식이며 Chroma score는 일반적으로
    낮을수록 더 유사한 거리 점수다.
    """
    results = vectordb.similarity_search_with_score(
        query=query,
        k=top_k
    )

    if results:
        best_score = results[0][1]
        logger.info(
            "[Retriever] query=%r top_k=%d returned=%d best_score=%.4f",
            query[:120],
            top_k,
            len(results),
            best_score,
        )
    else:
        logger.info("[Retriever] query=%r top_k=%d returned=0", query[:120], top_k)

    return results
