# U-sto_AI
대학물품관리시스템

1. Manual JSON → QA 변환
2. QA → Embedding 생성
3. ChromaDB 저장
4. Retriever 검색
5. LLM 응답 생성


본 시스템은 Chroma(VectorStore)를 사용하며,
ChromaDB의 HNSW 기반 벡터 인덱스를 통해
유사도 기반 검색을 수행한다.


검색은 similarity_search_with_score를 통해 수행되며,
거리 기반 벡터 유사도 계산 결과를 반환한다.