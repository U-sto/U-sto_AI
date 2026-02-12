# Chroma client 생성
# collection 생성
# 벡터 insert / persist 처리

from langchain_community.vectorstores import Chroma  # Chroma 벡터 DB
from langchain_core.documents import Document  # Document 타입
from typing import List  # 타입 힌트

def create_chroma_db(documents: List[Document], embeddings, persist_dir: str):
    # 문서 리스트를 Chroma DB로 저장하는 함수
    vectordb = Chroma.from_documents(
        documents=documents,              # Document 리스트
        embedding=embeddings,              # 임베딩 함수
        persist_directory=persist_dir      # 저장 경로
    )

    vectordb.persist()  # 디스크 저장 강제
    return vectordb     # DB 객체 반환

def load_chroma_db(embeddings, persist_dir: str):
    # 기존 Chroma DB 로딩 함수
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
