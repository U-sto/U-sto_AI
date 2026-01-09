 # env, path, model 설정
# - 모델 이름
# - 경로
# - 환경 변수
# - embedding/LLM 설정

"""
config.py
- 프로젝트 전역 설정 모음
- 로직 금지
"""

# ===============================
# 📂 경로 설정
# ===============================

# Chroma 벡터 DB 저장 경로
VECTOR_DB_PATH = "chroma_db"


# ===============================
# 🤖 LLM 설정
# ===============================

# 사용할 Chat 모델 이름
LLM_MODEL_NAME = "gpt-4o"

# LLM 응답 안정성 조절
LLM_TEMPERATURE = 0.1


# ===============================
# 🧠 Embedding 설정
# ===============================

# 임베딩 모델 이름
EMBEDDING_MODEL_NAME = "text-embedding-3-small"


# ===============================
# 🔍 Retriever 설정
# ===============================

# 검색 시 가져올 문서 개수
RETRIEVER_TOP_K = 30

# 유사도/거리 점수 threshold
# 값이 작을수록 더 유사하며, threshold 초과 문서는 폐기
SIMILARITY_SCORE_THRESHOLD = 10.0

# LLM에 전달할 최대 문서 수
TOP_N_CONTEXT = 6

# 누적 확률 기반 샘플링
Top_p = 0.9        

# # ===============================
# # 🔁 Re-ranking (Cross-Encoder)
# # ===============================

# # Cross-Encoder 모델 이름
# CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# # Re-ranking 이후 최종 context 개수
# RERANK_TOP_N = 8


# ===============================
# 🗣️ 프롬프트 관련 설정
# ===============================

# 참고 자료 없을 때 고정 응답
NO_CONTEXT_RESPONSE = (
    "죄송합니다, 매뉴얼에 해당 내용이 없어 답변드리기 어렵습니다."
)
