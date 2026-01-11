# JSON 파일 로딩 및 Chunking 역할
# dataset_sample의 원본 지식 읽기 담당

import json  # JSON 파싱 모듈
import os    # 파일 경로 처리 모듈
from typing import List, Dict  # 타입 힌트용

# LOOKBACK_RATIO: 청크 분할 시 문맥 유지를 위해 뒤로 되돌아가는 최대 비율 (30%)
# (이유: 너무 많이 뒤로 가면 청크 길이가 지나치게 짧아져 효율이 떨어짐)
LOOKBACK_RATIO = 0.3

# Chunking 규칙 함수 정의 (여기서 규칙을 수정합니다)
def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    긴 텍스트를 일정 길이의 청크로 나누되, 단어 중간이 잘리지 않도록
    뒤로 되짚어 보면서 자연스러운 분리 지점을 찾는 함수입니다.
    
    각 청크는 `chunk_size`를 최대 길이로 하여 생성되며, 인접한 청크 간에는
    `overlap` 길이만큼의 겹치는 구간을 유지해 앞/뒤 문맥이 이어지도록 합니다.
    
    Parameters
    ----------
    text : str
        분할할 원본 텍스트입니다.
    chunk_size : int, optional
        생성하려는 청크의 목표 최대 길이(문자 수)입니다.
        실제 청크 길이는 단어를 자르지 않기 위해 약간 더 짧아질 수 있습니다.
        `overlap`보다 반드시 커야 하며, 그렇지 않으면 예외가 발생합니다.
    overlap : int, optional
        인접한 청크 사이에서 앞선 청크의 끝 부분을 얼마나 겹쳐서 포함할지에
        대한 길이(문자 수)입니다. 이 값이 클수록 청크 간 문맥은 잘 유지되지만
        전체 텍스트 길이에 비해 청크 개수가 많아질 수 있습니다.
    
    Chunking strategy
    -----------------
    1. 기본적으로 `start` 위치에서 `chunk_size`만큼 떨어진 `end` 지점을
       후보 경계로 잡습니다.
    2. 마지막 청크가 아니라면, `end` 지점에서 최대 `overlap` 길이만큼
       뒤로 되짚어 보며 공백이나 마침표 등의 구두점을 찾습니다.
       - 발견되면 그 지점을 새로운 `end`로 사용해 단어/문장 중간이
         잘리지 않도록 합니다.
       - 발견되지 않으면 원래 `end`를 사용해 정확히 `chunk_size` 주변에서
         잘라냅니다.
    
    Infinite loop prevention
    ------------------------
    - 함수 시작 시 `chunk_size <= overlap`인 경우 즉시 `ValueError`를 발생시켜
      루프가 앞으로 진행되지 못하는 설정을 사전에 차단합니다.
    - 각 반복에서 다음 `start` 위치를 `end - overlap`으로 계산해
      청크 간에 지정된 만큼의 겹치는 구간을 유지합니다.
    - 만약 계산된 `next_step`이 현재 `start`와 같거나 더 앞이라면
      (예: 청크가 너무 짧게 잘린 경우), 겹침을 포기하고 `start = end`로
      이동시켜 최소한 한 번 이상 전진하도록 보장합니다.
    - 마지막으로, `while start < text_len:` 조건 자체가 `start`가
      `text_len`에 도달하거나 이를 초과하면 루프를 종료하도록 보장해
      예기치 못한 조건에서의 무한 루프를 방지합니다.
    
    Returns
    -------
    List[str]
        분할된 텍스트 청크들의 리스트입니다. 각 요소는 공백이 앞뒤에서
        제거된 문자열이며, 빈 문자열은 포함되지 않습니다.
    """
    
    
    # [Fix 1] 유효성 검사: chunk_size가 overlap보다 크지 않으면 루프 진행이 멈추거나 무한 루프가 발생할 수 있음
    # (각 반복에서 start가 end-overlap으로 이동하므로, chunk_size <= overlap이면 start가 증가하지 않음)
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size({chunk_size})는 overlap({overlap})보다 커야 합니다.")
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)

        # 마지막 부분이 아니라면, 단어가 잘리지 않게 뒤에서부터 공백/구두점 탐색
        if end < text_len:
            # end 지점에서부터 overlap 범위 내에서 가장 가까운 공백/마침표 찾기
            look_back_limit = max(start + 1, end - int(chunk_size * LOOKBACK_RATIO))

            found_split = False
            for i in range(end, look_back_limit, -1):
                if text[i] in [' ', '\n', '.', '!', '?']:
                    end = i + 1  # 공백/구두점 다음에서 자름
                    found_split = True
                    break
            # 위의 제한된 범위 내에서 적절한 분기점을 못 찾았다면,
            # 현재 청크(start~end) 전체를 대상으로 한 번 더 뒤에서부터 탐색
            if not found_split:
                for i in range(end, start, -1):
                    if text[i] in [' ', '\n', '.', '!', '?']:
                        end = i + 1  # 공백/구두점 다음에서 자름
                        found_split = True
                        break
            # 여전히 분기점을 찾지 못한 경우에는 기본 end에서 잘라 단어가
            # 중간에서 잘릴 수 있다는 점을 감수한다.

        chunk = text[start:end].strip()

        # 빈 청크가 아니면 추가
        if chunk:
            chunks.append(chunk)

        # [Fix 2] 다음 시작점 계산 로직 개선
        # 무한 루프 방지를 위해 반드시 start가 증가하도록 보장
        next_step = end - overlap

        # 겹치는 부분이 현재 시작점보다 뒤에 있어야 함.
        # 만약 청크가 너무 작아서 next_step이 start보다 앞서거나 같다면,
        # 겹치지 않더라도 그냥 end부터 시작하게 함 (무한 루프 방지)
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

                # 본문(content)을 제외한 나머지 메타데이터만 추출
                document_metadata = {k: v for k, v in item.items() if k != "content"}

                # 잘려진 조각들을 각각 별도의 문서로 저장
                # enumerate를 사용하여 chunk_index 생성 (기존 코드 버그 수정)
                for i, chunk in enumerate(text_chunks):
                    new_doc = document_metadata.copy()  # 원본 메타데이터 복사
                    new_doc["content"] = chunk          # 본문을 잘라진 조각으로 교체
                    new_doc["source"] = file_name       # 출처 기록
                    new_doc["chunk_index"] = i          # 청크 순서 기록 (0, 1, 2...)
                    
                    # [피드백 반영] doc_id 추가
                    # 형식: {파일명}{인덱스} (예: data.json0, data.json1)
                    new_doc["doc_id"] = f"{file_name}{i}"

                    documents.append(new_doc)      # 리스트에 추가

    return documents  # 로드된 전체 문서 반환