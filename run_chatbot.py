# 챗봇 실행 시도용 파일
import os
import sys
import io
import json
import re
from dotenv import load_dotenv

# LangChain 임포트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 화면 출력 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

load_dotenv()

# =========================================================
# [설정] Bi-gram(2글자) 매칭 하이브리드 검색 시스템
# =========================================================
SEARCH_K = 50 
DATA_FILE = "dataset/qa_output/manual_qa_final.json"
CHROMA_DB_PATH = "./chroma_db"

def initialize_system():
    print("🔄 시스템 초기화 중... (데이터 로드 및 검색기 준비)")
    
    if not os.path.exists(DATA_FILE):
        print("❌ 데이터 파일이 없습니다.")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        # 제목과 질문을 강조해서 내용 생성
        page_content = f"[{item.get('category')}] {item.get('title')}\nQ: {item.get('question')}\nA: {item.get('answer')}"
        metadata = {"source": item.get("source")}
        docs.append(Document(page_content=page_content, metadata=metadata))

    # 1. BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = SEARCH_K
    
    # 2. Chroma
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
    
    # 3. LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    print("✅ 시스템 준비 완료!")
    return bm25_retriever, chroma_retriever, llm

def get_bigrams(text):
    """문자열을 2글자씩 잘라서 리스트로 반환 (예: '반납의' -> ['반납', '납의'])"""
    text = re.sub(r"\s+", "", text) # 공백 제거
    return [text[i:i+2] for i in range(len(text) - 1)]

def calculate_match_score(query, doc_content):
    """
    [핵심 로직] Bi-gram 매칭 점수 계산
    질문의 2글자 조각들이 문서에 얼마나 많이 들어있는지 확인
    """
    query_bigrams = get_bigrams(query)
    doc_clean = re.sub(r"\s+", "", doc_content)
    
    score = 0
    matched_cnt = 0
    
    for bigram in query_bigrams:
        if bigram in doc_clean:
            matched_cnt += 1
            
            # 제목 부분(앞 50자)에 있으면 가산점 폭탄
            if bigram in doc_clean[:50]:
                score += 30.0 
            else:
                score += 5.0
    
    return score

def hybrid_search(query, bm25_retriever, chroma_retriever):
    # 1. 기본 검색 (BM25 + Chroma)
    bm25_res = bm25_retriever.invoke(query)
    chroma_res = chroma_retriever.invoke(query)
    
    score_map = {}
    
    # RRF (순위 기반 점수)
    for i, doc in enumerate(bm25_res):
        key = doc.page_content
        if key not in score_map: score_map[key] = {'doc': doc, 'score': 0}
        score_map[key]['score'] += (1.0 / (i + 1))

    for i, doc in enumerate(chroma_res):
        key = doc.page_content
        if key not in score_map: score_map[key] = {'doc': doc, 'score': 0}
        score_map[key]['score'] += (1.0 / (i + 1))

    # 2. [필살기] Bi-gram 매칭 점수 추가
    # 질문에 있는 단어 조각이 문서에 포함되어 있으면 점수를 팍팍 줍니다.
    for key, item in score_map.items():
        match_score = calculate_match_score(query, key)
        item['score'] += match_score

    # 3. 정렬 및 Top 3 추출
    sorted_items = sorted(score_map.values(), key=lambda x: x['score'], reverse=True)
    return [item['doc'] for item in sorted_items[:3]]

def generate_answer(query, docs, llm):
    if not docs:
        return "관련된 문서를 찾을 수 없습니다."

    context_text = "\n\n".join([f"문서 {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
    
    system_prompt = """
    당신은 사내 규정 전문가 챗봇입니다.
    [검색된 문서]를 바탕으로 답변하세요.
    
    1. 질문의 의도(정의, 절차, 방법 등)에 가장 적합한 문서를 우선적으로 참고하세요.
    2. '반납'에 대해 물어보면 '반납 개요'나 '반납 절차' 문서를 참고하여 설명하세요.
    3. 문서에 없는 내용은 "매뉴얼에 내용이 없습니다"라고 하세요.
    """
    
    user_prompt = f"""
    [검색된 문서]
    {context_text}

    [질문]
    {query}
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return response.content

def main():
    bm25, chroma, llm = initialize_system()
    
    print("=" * 60)
    print("💡 팁: '종료'를 입력하면 꺼집니다.")
    
    while True:
        try:
            user_input = input("\n🗣️  질문: ")
            
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("👋 안녕히 가세요!")
                break
            
            if not user_input.strip():
                continue

            print("   🔍 매뉴얼 정밀 검색 중 (Bi-gram)...")
            
            relevant_docs = hybrid_search(user_input, bm25, chroma)
            
            # [디버깅] 어떤 문서가 뽑혔는지 확인
            print(f"   👉 Top 1 문서: {relevant_docs[0].page_content.split('Q:')[0].strip()}")

            print("   🤖 답변 생성 중...")
            answer = generate_answer(user_input, relevant_docs, llm)
            
            print("\n📢 [AI 답변]")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    main()