import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 0. 데이터 로드 (Phase 2 결과물)
# ---------------------------------------------------------
print("📂 [Phase 3] 원천 데이터 로드 중...")

try:
    df_op = pd.read_csv('04_01_operation_master.csv') # 운용대장
    df_rt = pd.read_csv('04_03_return_list.csv')      # 반납
    df_du = pd.read_csv('05_01_disuse_list.csv')      # 불용
    df_dp = pd.read_csv('06_01_disposal_list.csv')    # 처분
except FileNotFoundError as e:
    print(f"❌ 오류: 파일이 없습니다. Phase 2를 먼저 실행해주세요. ({e})")
    exit()

# 날짜 컬럼 형변환 (검증용)
date_cols = ['취득일자', '정리일자']
for col in date_cols:
    if col in df_op.columns: df_op[col] = pd.to_datetime(df_op[col])

# ---------------------------------------------------------
# 1. 화면별 View CSV 생성
# ---------------------------------------------------------
print("⚙️ [Phase 3] 화면별 요구사항에 따른 View CSV 생성 중...")

# [04-03] 물품 반납 관리
# 1) 상단 그리드: 반납 등록 목록 (신청 건 위주)
#    (실제론 Request ID로 묶여야 하지만, 여기선 개별 행을 신청 건으로 간주)
view_rt_reg = df_rt[['반납일자', '반납확정일자', '등록자ID', '등록자명', '승인상태']]
view_rt_reg.to_csv('View_04_03_반납등록목록.csv', index=False, encoding='utf-8-sig')

# 2) 하단 그리드: 반납 물품 목록 (상세)
view_rt_item = df_rt[['G2B_목록번호', 'G2B_목록명', '물품고유번호', '취득일자', '취득금액', '운용부서', '물품상태', '사유']]
view_rt_item.to_csv('View_04_03_반납물품목록.csv', index=False, encoding='utf-8-sig')


# [05-01] 물품 불용 관리
# 1) 상단 그리드: 불용 등록 목록
view_du_reg = df_du[['불용일자', '불용확정일자', '등록자ID', '등록자명', '승인상태']]
view_du_reg.to_csv('View_05_01_불용등록목록.csv', index=False, encoding='utf-8-sig')

# 2) 하단 그리드: 불용 물품 목록
view_du_item = df_du[['G2B_목록번호', 'G2B_목록명', '물품고유번호', '취득일자', '취득금액', '운용부서', '물품상태', '사유']]
view_du_item.to_csv('View_05_01_불용물품목록.csv', index=False, encoding='utf-8-sig')


# [06-01] 물품 처분 관리
# 1) 상단 그리드: 처분 목록
view_dp_reg = df_dp[['처분일자', '처분정리구분', '등록자ID', '등록자명', '승인상태']]
view_dp_reg.to_csv('View_06_01_처분목록.csv', index=False, encoding='utf-8-sig')

# 2) 하단 그리드: 처분 물품 목록
# 요청하신 '정리일자', '불용일자', '내용연수' 포함
view_dp_item = df_dp[['G2B_목록번호', 'G2B_목록명', '물품고유번호', '취득일자', '취득금액', 
                      '처분방식', '물품상태', '사유', '정리일자', '불용일자', '내용연수']]
view_dp_item.to_csv('View_06_01_처분물품목록.csv', index=False, encoding='utf-8-sig')


# [07-01] 보유 현황 조회 (Aggregation)
# "수량" 컬럼이 필요하므로, 개별 자산을 그룹핑하여 카운트해야 함
# 기준: G2B번호, 취득일자, 부서, 상태 등이 같으면 같은 묶음으로 간주
print("   - 보유 현황 집계(Grouping) 중...")

# [핵심 수정] NaN 값을 빈 문자열로 치환하여 groupby 시 누락 방지
# 운용부서가 없는(반납/불용/처분) 데이터도 카운트에 포함되어야 함
df_op_filled = df_op.fillna("")

group_cols = ['G2B_목록번호', 'G2B_목록명', '취득일자', '취득금액', '정리일자', 
              '운용부서', '운용상태', '내용연수']

# dropna=False 옵션은 pandas 최신 버전 기능이므로, 위에서 fillna를 하는 방식이 가장 안전함
view_inventory = df_op_filled.groupby(group_cols).size().reset_index(name='수량')
view_inventory.to_csv('View_07_01_보유현황.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------
# 2. 데이터 정합성 검증 (Validation)
# ---------------------------------------------------------
print("\n🔍 [Phase 3] 데이터 정합성 검증 시작")

# 검증 1: 수량 일치 여부
# 운용대장의 전체 행 수 vs 보유현황의 수량 총합
total_assets_op = len(df_op)
total_assets_inv = view_inventory['수량'].sum()

print(f"1. 수량 검증: 운용대장({total_assets_op}) vs 보유현황합계({total_assets_inv})")
if total_assets_op == total_assets_inv:
    print("   ✅ PASS: 총 수량이 정확히 일치합니다.")
else:
    print("   ❌ FAIL: 수량이 불일치합니다.")

# 검증 2: 날짜 논리 확인 (불용확정일자 > 취득일자)
# 불용 목록에서 샘플링하여 확인
print("2. 날짜 논리 검증 (불용일자 > 취득일자)")
error_count = 0
for _, row in df_du.iterrows():
    acq_d = pd.to_datetime(row['취득일자'])
    du_d = pd.to_datetime(row['불용일자'])
    if du_d < acq_d:
        error_count += 1

if error_count == 0:
    print("   ✅ PASS: 모든 데이터가 시간 순서(취득->불용)를 준수합니다.")
else:
    print(f"   ❌ FAIL: {error_count}건의 데이터에서 시간 역전 현상 발생.")

# 검증 3: 반려 데이터 격리 확인
# 운용대장에는 '확정'된 건만 있어야 하므로, 
# 만약 반납/불용/처분에서 '반려'된 건이 운용상태를 변경시켰는지 확인
# (Phase 2 로직상 반려되면 상태 변경 안 함 -> 운용대장엔 이전 상태로 남아있어야 함)
# 여기서는 간단히 처분 리스트의 '승인상태'가 '확정'인 것만 '처분' 상태인지 확인
print("3. 상태 로직 검증 (처분 확정 건)")
disposal_confirmed_ids = df_dp[df_dp['승인상태'] == '확정']['물품고유번호'].tolist()
op_disposed_rows = df_op[df_op['물품고유번호'].isin(disposal_confirmed_ids)]
non_disposed_status = op_disposed_rows[op_disposed_rows['운용상태'] != '처분']

if len(non_disposed_status) == 0:
    print("   ✅ PASS: 처분 확정된 모든 물품의 운용상태가 '처분'으로 변경되었습니다.")
else:
    print(f"   ❌ FAIL: 처분 확정되었으나 상태가 변경되지 않은 건이 {len(non_disposed_status)}개 있습니다.")

print("\n🎉 모든 작업이 완료되었습니다.")
print("   생성된 파일 목록:")
print("   - View_04_03_반납등록목록.csv / View_04_03_반납물품목록.csv")
print("   - View_05_01_불용등록목록.csv / View_05_01_불용물품목록.csv")
print("   - View_06_01_처분목록.csv / View_06_01_처분물품목록.csv")
print("   - View_07_01_보유현황.csv")