import os
import json
import glob
from openai import OpenAI
# 이유

# - 데이터 생성 파이프라인
# - 서비스 런타임과 분리
# ============================
# 1. API 키 설정 (환경 변수 또는 직접 입력)
# 실제 키를 입력하거나, 시스템 환경 변수에 설정되어 있어야 합니다.
API_KEY = os.getenv("OPENAI_API_KEY")

# (혹시나 해서 넣는 안전장치)
if not API_KEY:
    print("❌ 오류: .env 파일을 찾을 수 없거나 키가 없습니다!")

client = OpenAI(api_key=API_KEY)

def generate_qa_pairs(context_text, model="gpt-4o"):
    """
    주어진 매뉴얼 텍스트(context)를 바탕으로
    사용자가 물어볼 법한 질문(Q)과 그에 대한 답변(A)을 3~5개 생성합니다.
    """
    
    prompt = f"""
    아래는 [대학 물품 관리 시스템 매뉴얼]의 일부입니다.
    이 내용을 학습 데이터로 쓰기 위해, 사용자가 할 법한 '질문(Q)'과 그에 맞는 '답변(A)' 쌍을 3개에서 5개 정도 생성해주세요.

    [매뉴얼 내용]
    {context_text}

    [작성 가이드]
    1. 질문은 초보자, 실무자, 관리자가 할 수 있는 다양한 관점에서 만들어주세요.
    2. 답변은 매뉴얼 내용에 근거하여 정확하고 친절하게 작성해주세요.
    3. 단순한 질문뿐만 아니라, "A와 B의 차이는?", "이럴 땐 어떻게 해?" 같은 상황 질문도 포함해주세요.
    4. 출력 형식은 반드시 JSON Array 형태로만 주세요. 다른 말은 붙이지 마세요.
    
    [출력 예시]
    [
      {{"question": "물품 반납은 어떻게 하나요?", "answer": "운용 부서에서 반납 등록 및 승인 요청을 하고, 관리자가 확정하면 처리됩니다."}},
      {{"question": "불용과 처분의 차이가 뭔가요?", "answer": "불용은 사용 중단을 결정하는 행정 절차이고, 처분은 실제로 매각하거나 폐기하여 자산을 없애는 실행 단계입니다."}}
    ]
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates QA datasets in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # 응답 텍스트에서 JSON 부분만 파싱
        content = response.choices[0].message.content
        # 혹시 모를 코드 블록 마크다운 제거 (```json ... ```)
        content = content.replace("```json", "").replace("```", "").strip()
        
        return json.loads(content)

    except Exception as e:
        print(f"Error generating QA: {e}")
        return []

def main():
    # 경로 설정
    input_folder = 'dataset/input'
    output_folder = 'dataset/output'
    output_file = os.path.join(output_folder, 'train_dataset_final.json')

    # 출력 폴더 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_qa_data = []
    
    # manual_chapter로 시작하는 모든 json 파일 찾기
    input_files = glob.glob(os.path.join(input_folder, 'manual_chapter*.json'))
    
    # 파일명 순서대로 정렬 (1, 2, 3... 순서대로 처리하기 위해)
    input_files.sort()

    print(f"총 {len(input_files)}개의 매뉴얼 파일을 찾았습니다.")

    for file_path in input_files:
        print(f"📂 처리 중: {os.path.basename(file_path)}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 각 챕터의 섹션(청크)별로 순회
            for section in data:
                context = f"챕터: {section['chapter']} {section['title']}\n내용: {section['content']}"
                
                # AI에게 QA 생성 요청
                qa_pairs = generate_qa_pairs(context)
                
                # 결과 저장 (원본 출처도 함께 기록하면 나중에 디버깅하기 좋습니다)
                for qa in qa_pairs:
                    qa['source_chapter'] = section['chapter']
                    qa['source_title'] = section['title']
                    # RAG 검색에 쓰일 원본 텍스트(Context)도 같이 저장하는 경우가 많습니다.
                    qa['context'] = section['content'] 
                    all_qa_data.append(qa)
                    
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({file_path}): {e}")

    # 최종 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_data, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print(f"✅ 모든 작업 완료! 총 {len(all_qa_data)}개의 QA 데이터가 생성되었습니다.")
    print(f"💾 저장 위치: {output_file}")

if __name__ == "__main__":
    main()