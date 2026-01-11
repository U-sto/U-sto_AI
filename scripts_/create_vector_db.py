# scripts_/create_vector_db.py

import os
import sys
import io
import json
import shutil
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================================
# [화면 출력 인코딩 설정]
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
# ==========================================

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("오류: .env 파일이 없거나 OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

def main():
    # [설정] 경로 지정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # [중요] 타겟 파일 경로 (generate_qa.py의 결과물)
    TARGET_FILE = os.path.join(root_dir, 'dataset/qa_output/manual_qa_final.json')
    
    # DB 저장 경로
    DB_PATH = os.path.join(root_dir, 'chroma_db')

    print("벡터 DB 생성 작업을 시작합니다...")

    # 1. 기존 DB 삭제 (초기화)
    if os.path.exists(DB_PATH):
        print(f"기존 DB 폴더('{DB_PATH}')를 초기화합니다...")
        try:
            shutil.rmtree(DB_PATH)
        except Exception as e:
            print(f"경고: 기존 DB 삭제 중 오류 발생 (무시하고 진행합니다): {e}")

    # 2. 데이터 로드
    if not os.path.exists(TARGET_FILE):
        print(f"오류: 파일이 없습니다 -> {TARGET_FILE}")
        print("   먼저 generate_qa.py를 실행해서 데이터를 만들어주세요.")
        sys.exit(1)

    print(f"타겟 파일 로드 중: {os.path.basename(TARGET_FILE)}")
    
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON 파일 읽기 실패: {e}")
        sys.exit(1)

    # 데이터가 리스트가 아니면 즉시 종료
    if not isinstance(data, list):
        print("[오류] 데이터 형식이 잘못되었습니다.")
        print(f" - 기대 형식: list (JSON Array)")
        print(f" - 실제 형식: {type(data)}")
        sys.exit(1)

    # 3. LangChain Document 객체로 변환
    documents = []
    
    if isinstance(data, list):
        # [Fix] enumerate를 추가하여 idx(순서 번호)를 생성합니다!
        for idx, item in enumerate(data):
            # 질문(Q)과 답변(A) 가져오기
            q = item.get('question', '')
            a = item.get('answer', '')
            
            # 둘 다 내용이 있을 때만 처리
            if q and a:
                # 검색 내용 구성
                content = f"Category: {item.get('category', 'General')}\nTitle: {item.get('title', '')}\nQ: {q}\nA: {a}"
                
                # 메타데이터 연결
                metadata = {
                    "source": item.get("source", "Unknown"),
                    "title": item.get("title", ""),
                    "chapter": item.get("chapter", ""),
                    "category": item.get("category", "General"),
                    # [핵심] 이제 idx가 존재하므로 여기서 에러가 나지 않습니다.
                    # doc_id가 없으면 '파일명_번호' 형식으로 생성
                    "doc_id": item.get("doc_id", f"{item.get('source', 'doc')}_{idx}")
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
    else:
        print("경고: JSON 데이터가 리스트 형식이 아닙니다.")

    # 변환된 데이터가 하나도 없는 경우 처리
    if not documents:
        print("변환할 데이터가 없습니다.")
        sys.exit(1)

    print(f"총 {len(documents)}개의 지식(QA)을 준비했습니다.")

    # 4. 임베딩 및 DB 저장
    print("Embedding 모델을 준비 중입니다...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 임베딩 진행 상황 구체적 명시
    print(f"Chroma DB 저장 시작... (총 {len(documents)}개 벡터 변환, 저장 위치: {DB_PATH})")
    print("(데이터 양에 따라 시간이 조금 걸릴 수 있습니다...)")
    
    try:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
    except Exception as e:
        print(f"[Fatal Error] 임베딩 및 DB 저장 중 실패: {e}")
        sys.exit(1)

    print("-" * 30)
    print("DB 생성 완료!")
    print(f"이제 '{DB_PATH}' 폴더에 AI의 지식이 저장되었습니다.")

if __name__ == "__main__":
    main()