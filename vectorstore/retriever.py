# query embedding 생성
# similarity search
# top-k 문서 반환

from typing import List, Tuple

def retrieve_docs(vectordb, query: str, top_k: int) -> List[Tuple]:
    """
Chroma VectorStore를 통해 유사 문서 검색 수행
내부적으로 ChromaDB의 HNSW 기반 벡터 인덱싱 사용
"""


    # similarity_search_with_score
    # 반환값: [(Document, score), ...]
    results = vectordb.similarity_search_with_score(
        query=query,   # 사용자 질의 (문자열)
        k=top_k        # 충분히 큰 후보군
    )

    return results
