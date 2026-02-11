#  # 벡터 생성 로직

# QA 텍스트 → 벡터 변환
# OpenAI / HuggingFace embedding 연결 지점

from langchain_openai import OpenAIEmbeddings  # 임베딩 클래스
from app.config import EMBEDDING_MODEL_NAME

def get_embedding_model():
    # 임베딩 모델 생성 함수
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
