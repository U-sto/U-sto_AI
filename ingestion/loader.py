# JSON 파일 로딩 역할
# dataset_sample의 원본 지식 읽기 담당

import json  # JSON 파싱 모듈
import os    # 파일 경로 처리 모듈
from typing import List, Dict  # 타입 힌트용

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

            # 각 아이템에 source 정보 추가
            for item in data:
                item["source"] = file_name  # 출처 파일명 기록
                documents.append(item)      # 리스트에 추가

    return documents  # 로드된 전체 문서 반환
