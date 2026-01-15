# 추후 삭제할 파일입니다.
# test_tools.py
from rag.tools import get_asset_basic_info, navigate_to_page, open_usage_prediction_page

def main():
    print("--------------------------------------------------")
    print("🛠️  Function Calling 도구 테스트 시작")
    print("--------------------------------------------------")

    # 1. 물품 조회 테스트 (Mock 데이터에 있는 '맥북' 검색)
    print("\n[Test 1] 물품 조회: '맥북 프로'")
    result1 = get_asset_basic_info(asset_name="맥북 프로")
    print(f"▶ 결과: {result1}")

    # 2. 물품 조회 테스트 (없는 물품 검색)
    print("\n[Test 2] 물품 조회: '없는 물건'")
    result2 = get_asset_basic_info(asset_name="투명망토")
    print(f"▶ 결과: {result2}")

    # 3. 화면 이동 테스트
    print("\n[Test 3] 화면 이동: 'ASSET_DETAIL'")
    result3 = navigate_to_page(page_type="ASSET_DETAIL")
    print(f"▶ 결과: {result3}")
    
    # 4. 예측 페이지 연결 테스트
    print("\n[Test 4] 예측 페이지: 키워드 '에어컨'")
    result4 = open_usage_prediction_page(keyword="에어컨")
    print(f"▶ 결과: {result4}")

    print("\n--------------------------------------------------")
    print("✅ 테스트 완료! 결과가 JSON 형태로 잘 나오면 성공입니다.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()