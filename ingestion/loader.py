# JSON 파일 로딩 및 Chunking 역할
# dataset_sample의 원본 지식 읽기 담당

import json  # JSON 파싱 모듈
import os    # 파일 경로 처리 모듈
from typing import List, Dict  # 타입 힌트용

# Chunking 규칙 함수 정의 (여기서 규칙을 수정합니다)
def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len and text[end] != " ":
            last_space = text.rfind(" ", start, end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_step = end - overlap
        if next_step <= start: 
            start = end
        else:
            start = next_step
        
    return chunks

# 메인 로더 함수
def load_json_files(folder_path: str) -> List[Dict]:
    # 지정된 폴더 내 JSON 파일들을 모두 로드하는 함수
    documents = []  # 결과 저장 리스트 초기화

    # 폴더 내 파일 순회
    for file_name in os.listdir(folder_path):
        # JSON 파일만 처리
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file_name)  # 전체 경로 생성

        # 파일 열기
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # JSON 로드

            # 각 아이템 순회
            for item in data:
                origin_text = item.get("content", "") 
                
                # 텍스트가 비어있으면 스킵
                if not origin_text:
                    continue

                # 위에서 만든 규칙대로 텍스트 자르기 (Chunking)
                text_chunks = split_text(origin_text, chunk_size=500, overlap=50)

                # 잘라진 조각들을 각각 별도의 문서로 저장
                for chunk in text_chunks:
                    new_doc = item.copy()  # 원본 메타데이터 복사
                    new_doc["content"] = chunk # 본문을 잘라진 조각으로 교체
                    new_doc["source"] = file_name # 출처 기록
                    
                    documents.append(new_doc)      # 리스트에 추가

    return documents  # 로드된 전체 문서 반환