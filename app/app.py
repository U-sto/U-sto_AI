# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# # 1. 파일 경로 설정 (정의부터 확실히!)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# env_path = os.path.join(current_dir, '.env')

# # 2. .env 파일 로드 시도
# load_dotenv(env_path)

# # 3. 키 가져오기
# api_key = os.getenv("OPENAI_API_KEY")

# # 4. 안전장치 (NameError 방지 및 에러 메시지 통합)
# if not api_key:
#     # st.set_page_config(page_title="설정 오류", page_icon="🚨")
#     st.error("🚨 **API 키를 로드할 수 없습니다!**")
#     st.info(f"📍 확인 중인 경로: {env_path}")
#     st.warning("**.env 파일 확인 리스트:**\n"
#                "1. 파일 이름이 정확히 `.env` 인가요? (`.env.txt` 아님)\n"
#                "2. 파일 안에 `OPENAI_API_KEY=sk-...` 라고 적었나요?\n"
#                "3. 등호(=) 앞뒤에 공백은 없나요?")
#     st.stop() 

# # 5. 성공 시에만 환경변수 설정
# os.environ["OPENAI_API_KEY"] = api_key

# ###

# # 1. 페이지 기본 설정
# st.set_page_config(page_title="대학 물품 관리 AI", page_icon="🎓")
# st.title("🎓 대학 물품 관리 시스템 AI 챗봇")
# st.caption("🚀 이제 매뉴얼 내용을 기반으로 정확하게 답변합니다.")

# # 2. 진짜 데이터(DB) 로드 함수
# @st.cache_resource
# def get_qa_chain():
#     # ⚠️ 중요: DB 만들 때 썼던 모델과 똑같은 걸 써야 합니다.
#     embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
#     # 방금 만든 'chroma_db' 폴더와 연결!
#     persist_directory = 'chroma_db'
    
#     if not os.path.exists(persist_directory):
#         st.error("❌ 'chroma_db' 폴더를 찾을 수 없습니다. create_vector_db.py를 먼저 실행했나요?")
#         return None

#     # DB 로드
#     vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

#     retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
#     # 똑똑한 GPT-4o 연결
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

#     # 프롬프트(지시사항) 설정
#     template = """
#     당신은 대학 물품 관리 시스템의 친절한 AI 도우미입니다.
#     아래 [참고 자료]를 바탕으로 질문에 답변해주세요.
#     자료에 없는 내용은 "죄송합니다, 매뉴얼에 없는 내용입니다."라고 솔직하게 말해주세요.

#     [참고 자료]:
#     {context}

#     질문: {question}
#     답변:
#     """
#     prompt = PromptTemplate.from_template(template)

#     chain = (
#         {
#             "context": retriever,
#             "question": lambda x: x
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain

# # 3. 채팅 UI 구성
# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 물품 관리에 대해 무엇이든 물어보세요."}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # 4. 질문 입력 처리
# if query := st.chat_input("질문을 입력하세요..."):
#     # 사용자 질문 표시
#     st.session_state.messages.append({"role": "user", "content": query})
#     st.chat_message("user").write(query)

#     # AI 답변 생성
#     with st.chat_message("assistant"):
#         chain = get_qa_chain()
#         if chain:
#             with st.spinner("매뉴얼을 찾아보는 중입니다... 📚"):
#                 try:
#                     # 답변 요청
#                     response_text = chain.invoke(query)
                    
#                     # # 출처 정리 (중복 제거)
#                     # sources = set([doc.metadata.get('source', '알 수 없음') for doc in source_docs])
                    
#                     # 화면 출력
#                     st.write(response_text)
                    
#                     # # 출처 표시 (작게)
#                     # if sources:
#                     #     st.caption(f"📚 참고 문서: {', '.join(sources)}")
                    
#                     # 기록 저장
#                     st.session_state.messages.append({"role": "assistant", "content": response_text})
                
#                 except Exception as e:
#                     st.error(f"오류가 발생했습니다: {e}")