# 추후 삭제할 파일입니다.
import os
import sys
import io
import json
import re
import difflib
from dotenv import load_dotenv

# [수정된 부분] 최신 LangChain 버전 호환 경로
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage 

# [화면 출력 설정]
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

load_dotenv()

# =========================================================
# [설정] 검색 + LLM 답변 생성 테스트
# =========================================================
SEARCH_K = 50 
DATA_FILE = "dataset/qa_output/manual_qa_final.json"
CHROMA_DB_PATH = "./chroma_db"

def safe_import():
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        return OpenAIEmbeddings, Chroma, BM25Retriever, Document
    except ImportError as e:
        print(f"❌ 라이브러리 에러: {e}")
        sys.exit(1)

def extract_question_from_text(text):
    match = re.search(r"Q:\s*(.*?)(?:\n|$)", text)
    return match.group(1).strip() if match else ""

def calculate_fuzzy_score(query, target):
    if not target: return 0.0
    norm_query = re.sub(r"\s+", "", query)
    norm_target = re.sub(r"\s+", "", target)
    matcher = difflib.SequenceMatcher(None, norm_query, norm_target)
    return matcher.ratio()

def main():
    print("🚀 [최종] 검색(Top 3) -> LLM 답변 생성 테스트\n")
    
    # 1. 검색 준비
    OpenAIEmbeddings, Chroma, BM25Retriever, Document = safe_import()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        page_content = f"[{item.get('category')}] {item.get('title')}\nQ: {item.get('question')}\nA: {item.get('answer')}"
        metadata = {"source": item.get("source")}
        docs.append(Document(page_content=page_content, metadata=metadata))

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = SEARCH_K
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})

    # 2. 질문
    user_query = "물품 반납 절차가 어떻게 돼?"
    print(f"👤 사용자 질문: '{user_query}'")
    print("-" * 50)

    # 3. 검색 실행 (Hybrid + Fuzzy)
    bm25_res = bm25_retriever.invoke(user_query)
    chroma_res = chroma_retriever.invoke(user_query)
    
    score_map = {}
    # BM25 점수
    for i, doc in enumerate(bm25_res):
        key = doc.page_content
        if key not in score_map: score_map[key] = {'doc': doc, 'score': 0}
        score_map[key]['score'] += (1.0/(i+1))

    # Chroma 점수
    for i, doc in enumerate(chroma_res):
        key = doc.page_content
        if key not in score_map: score_map[key] = {'doc': doc, 'score': 0}
        score_map[key]['score'] += (1.0/(i+1))

    # Fuzzy Bonus
    for key, item in score_map.items():
        doc_q = extract_question_from_text(key)
        if calculate_fuzzy_score(user_query, doc_q) >= 0.4:
            item['score'] += 10.0

    sorted_items = sorted(score_map.values(), key=lambda x: x['score'], reverse=True)
    top_3_docs = [item['doc'] for item in sorted_items[:3]]

    # 4. LLM에게 답변 요청
    print("🤖 LLM이 답변을 생각 중입니다...\n")
    
    context_text = "\n\n".join([f"문서 {i+1}:\n{d.page_content}" for i, d in enumerate(top_3_docs)])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 
    
    system_prompt = """
    당신은 사내 규정 챗봇입니다. 
    아래 제공된 [검색된 문서]를 바탕으로 질문에 답변하세요.
    검색된 문서에 정답이 있다면, 그 내용을 바탕으로 친절하게 설명하세요.
    """
    
    user_prompt = f"""
    [검색된 문서]
    {context_text}

    [질문]
    {user_query}
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    print("📢 [AI 답변 결과]")
    print("=" * 60)
    print(response.content)
    print("=" * 60)
    
    # 5. 검증
    if "운용 부서에서 더 이상 사용하지 않거나" in response.content or "고장" in response.content:
        print("\n✅ 성공! LLM이 올바른 문서를 참조하여 정답을 말했습니다.")
    else:
        print("\n⚠️ 답변 내용을 확인해보세요.")

if __name__ == "__main__":
    main()