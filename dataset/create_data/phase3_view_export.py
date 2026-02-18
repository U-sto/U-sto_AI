import pandas as pd
import os
from pandas.errors import EmptyDataError  # 에러 처리를 위해 import

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.join(BASE_DIR, "data_lifecycle") # 원천 데이터
SAVE_DIR = os.path.join(BASE_DIR, "data_view")      # 뷰 데이터 (create_data/data_view)
os.makedirs(SAVE_DIR, exist_ok=True) # data_view 폴더 생성

# 현재 유효한 상태를 의미하는 종료일 (무기한 유효)
CURRENT_STATUS_END_DATE = pd.Timestamp('2099-12-31')

# ---------------------------------------------------------
# 0. 데이터 로드
# ---------------------------------------------------------
print("📂 [Phase 3] 원천 데이터 로드 중...")

def safe_read_csv(file_path):
    """파일이 존재하고 비어있지 않을 때만 로드, 실패 시 빈 DF 반환"""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except EmptyDataError:
        return pd.DataFrame()

try:
    # Phase 2 결과물
    df_op = safe_read_csv(os.path.join(LOAD_DIR, '04_01_operation_master.csv'))
    df_rt = safe_read_csv(os.path.join(LOAD_DIR, '04_03_return_list.csv'))
    df_du = safe_read_csv(os.path.join(LOAD_DIR, '05_01_disuse_list.csv'))
    df_dp = safe_read_csv(os.path.join(LOAD_DIR, '06_01_disposal_list.csv'))
    df_hist = safe_read_csv(os.path.join(LOAD_DIR, '99_asset_status_history.csv'))
    
    # 운용 대장은 필수이므로 비어있으면 종료
    if df_op.empty:
        print("❌ 오류: '04_01_operation_master.csv' 데이터가 없거나 비어있습니다.")
        exit()

    print(f"   - 운용 마스터: {len(df_op)}건")

    # 리뷰어 지적 사항: 과거 부서 정보 복원을 위해 운용신청 이력 로드
    path_req = os.path.join(LOAD_DIR, '04_02_operation_req_list.csv')
    if os.path.exists(path_req):
        df_req = safe_read_csv(path_req)
    else:
        df_req = pd.DataFrame(columns=['물품고유번호', '운용부서', '운용신청일자'])
        
    # 데이터 프레임 전체의 NaN(결측치)를 빈 문자열로 치환 (문자열 컬럼만)
    # 날짜나 숫자는 그대로 두어야 오류가 안 남
    str_cols = ['비고', '운용부서', '운용상태', '승인상태', '사유', '물품상태']
    
    for df in [df_op, df_rt, df_du, df_dp]:
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')

except FileNotFoundError as e:
    print(f"❌ 오류: 파일이 없습니다. Phase 1, 2를 먼저 실행해주세요. ({e})")
    exit()
except Exception as e:
    print(f"❌ CSV 로드 중 알 수 없는 오류: {e}")
    exit()

# ---------------------------------------------------------
# 1. 화면별 View CSV 생성
# ---------------------------------------------------------
print("⚙️ [Phase 3] 화면별 요구사항에 따른 View CSV 생성 중...")

# [04-01] 물품 운용 - 물품기본정보 (Grouped View)
print("   - [04-01] 운용 화면용 기본정보 집계 중...")

group_cols_op = [
    'G2B_목록번호', 'G2B_목록명', '캠퍼스','취득일자', '취득금액', '정리일자', 
    '운용부서', '운용상태', '내용연수', '승인상태', 
    '취득정리구분', '운용부서코드', '비고'
]

# 데이터 컬럼 존재 확인
if set(group_cols_op).issubset(df_op.columns):
    view_op_basic = df_op.groupby(group_cols_op).size().reset_index(name='수량')
    view_op_basic.to_csv(os.path.join(SAVE_DIR, 'View_04_01_운용_기본정보.csv'), index=False, encoding='utf-8-sig')
else:
    print("   ⚠️ 경고: 04_01 파일에 필요한 컬럼이 부족합니다.")

# [06-01] 불용 물품 목록
# 병합에 사용할 Master 정보
master_cols = ['물품고유번호', '내용연수', '취득금액', '취득일자', '정리일자', 'G2B_목록명']
# df_du가 비어있거나 기본 키 컬럼이 없으면 병합 대신 빈 결과를 생성
if df_du.empty or '물품고유번호' not in df_du.columns or not set(master_cols).issubset(df_op.columns):
    print("   ⚠️ 경고: 06_01 파일 생성을 위한 데이터(df_du 또는 df_op)가 비어있거나 필요한 컬럼이 부족합니다.")
    base_cols = list(df_du.columns)
    # 병합 시 추가될 master_cols 중 기본 키는 제외하고, 아직 없는 컬럼만 추가
    extra_cols = [c for c in master_cols if c != '물품고유번호' and c not in base_cols]
    view_du_item = pd.DataFrame(columns=base_cols + extra_cols)
else:
    df_master_info = df_op[master_cols].drop_duplicates(subset=['물품고유번호'])
    cols_to_merge = [c for c in master_cols if c == '물품고유번호' or c not in df_du.columns]
    view_du_item = pd.merge(df_du, df_master_info[cols_to_merge], on='물품고유번호', how='left')

view_du_item.to_csv(os.path.join(SAVE_DIR, 'View_06_01_불용물품목록.csv'), index=False, encoding='utf-8-sig')

# [07-01] 보유 현황 조회 (SCD Type 2 History)
print("   - [07-01] 보유 현황(과거 시점 조회용) 데이터 생성 중...")

# [Fix] 리뷰 반영: df_hist 데이터 유효성 검증 (KeyError 방지)
required_hist_cols = ['물품고유번호', '변경일자', '(변경)운용상태']
group_cols_scd = [
    'G2B_목록번호', 'G2B_목록명', '캠퍼스',
    '취득일자', '취득금액', '정리일자', 
    '운용부서', '운용상태', '내용연수', '승인상태', 
    '취득정리구분', '운용부서코드', '비고',
    '유효시작일자', '유효종료일자'
]

# 가드 처리: 데이터가 비어있거나 필수 컬럼이 없는 경우 빈 스키마 생성
if df_hist.empty or not set(required_hist_cols).issubset(df_hist.columns):
    print("     ⚠️ 경고: 이력 데이터(df_hist)가 없거나 필수 컬럼이 누락되어 빈 데이터를 생성합니다.")
    view_inventory_scd = pd.DataFrame(columns=group_cols_scd + ['수량'])

else:
    try:
        # 1. 이력 데이터 정렬
        df_hist_proc = df_hist.copy()
        df_hist_proc['변경일자'] = pd.to_datetime(df_hist_proc['변경일자'], errors='coerce')
        # 날짜 변환 실패 행 제거 (NaT)
        df_hist_proc = df_hist_proc.dropna(subset=['변경일자'])
        df_hist_proc = df_hist_proc.sort_values(by=['물품고유번호', '변경일자'])

        # 2. 유효 기간(Start ~ End) 생성
        df_hist_proc['유효시작일자'] = df_hist_proc['변경일자']
        df_hist_proc['유효종료일자'] = df_hist_proc.groupby('물품고유번호')['변경일자'].shift(-1) - pd.Timedelta(days=1)
        df_hist_proc['유효종료일자'] = df_hist_proc['유효종료일자'].fillna(CURRENT_STATUS_END_DATE)

        # 3. 속성 정보 결합
        # 정적 정보(운용부서 포함)는 모두 df_op에서 가져옴
        static_cols = [
            'G2B_목록번호', 'G2B_목록명', '물품고유번호', '캠퍼스', '취득일자', '취득금액', '정리일자', 
            '내용연수', '승인상태', '취득정리구분','운용부서', '운용부서코드', '비고'
        ]
        # 실제 존재하는 컬럼만 선택하여 병합
        avail_static_cols = [c for c in static_cols if c in df_op.columns]
        df_static = df_op[avail_static_cols].drop_duplicates(subset=['물품고유번호'])
        
        df_scd_raw = pd.merge(df_hist_proc, df_static, on='물품고유번호', how='left')

        # 3-2. 부서 정보 복원 (리뷰 반영)
        if not df_req.empty and {'물품고유번호', '운용부서', '운용신청일자'}.issubset(df_req.columns):
            dept_map = (
                df_req.sort_values('운용신청일자')
                .drop_duplicates('물품고유번호', keep='last')[['물품고유번호', '운용부서']]
            )

            df_scd_raw = pd.merge(
                df_scd_raw,
                dept_map,
                on='물품고유번호',
                how='left',
                suffixes=('', '_req')
            )

            # 운용부서_req 컬럼이 존재할 때만 보정
            if '운용부서_req' in df_scd_raw.columns:
                if '운용부서' in df_scd_raw.columns:
                    df_scd_raw['운용부서'] = (
                        df_scd_raw['운용부서']
                        .replace('', pd.NA)
                        .fillna(df_scd_raw['운용부서_req'])
                        .fillna('')
                    )
                else:
                    df_scd_raw['운용부서'] = df_scd_raw['운용부서_req'].fillna('')
                
                df_scd_raw = df_scd_raw.drop(columns=['운용부서_req'], errors='ignore')

        # 4. 상태값 매핑 및 포맷팅
        df_scd_raw['운용상태'] = df_scd_raw['(변경)운용상태']
        df_scd_raw['유효시작일자'] = df_scd_raw['유효시작일자'].dt.strftime('%Y-%m-%d')
        df_scd_raw['유효종료일자'] = df_scd_raw['유효종료일자'].dt.strftime('%Y-%m-%d')
        df_scd_raw = df_scd_raw.fillna('')

        # 5. 그룹핑 및 수량 집계
        # 실제 데이터프레임에 존재하는 컬럼으로만 그룹핑 (KeyError 방지)
        actual_group_cols = [c for c in group_cols_scd if c in df_scd_raw.columns]
        view_inventory_scd = df_scd_raw.groupby(actual_group_cols).size().reset_index(name='수량')
        
    except Exception as e:
        print(f"     ❌ 이력 데이터 처리 중 오류 발생: {e}")
        view_inventory_scd = pd.DataFrame(columns=group_cols_scd + ['수량'])

view_inventory_scd.to_csv(os.path.join(SAVE_DIR, 'View_07_01_보유현황_이력기반.csv'), index=False, encoding='utf-8-sig')
# ---------------------------------------------------------
# SCD (이력 추적) 데이터 생성
# ---------------------------------------------------------
print("     SCD (Slowly Changing Dimension) 생성...")

# [Copilot Fix] 이력 데이터가 비어있을 경우 예외 처리
if df_hist.empty:
    print("      ⚠️ 이력 데이터가 없어 SCD 생성을 건너뜁니다.")
    # 빈 DataFrame 생성 (저장 시 에러 방지용 컬럼 정의)
    cols_scd = ['물품고유번호', 'Start_Date', 'End_Date', 'Is_Current', '운용상태', '사유', '관리자']
    df_scd = pd.DataFrame(columns=cols_scd)

else:
    # 1. 이력 데이터 정렬
    # '변경일자' 컬럼이 있는지 확인 후 변환
    if '변경일자' in df_hist.columns:
        df_hist['변경일자'] = pd.to_datetime(df_hist['변경일자'], errors='coerce')
        df_hist = df_hist.dropna(subset=['변경일자'])
        df_hist = df_hist.sort_values(by=['물품고유번호', '변경일자'])
        
        scd_list = []

        for asset_id, group in df_hist.groupby('물품고유번호'):
            group = group.sort_values('변경일자')
            prev_row = None
            
            for idx, row in group.iterrows():
                curr_date = row['변경일자']
                
                # 이전 이력 닫기
                if prev_row is not None:
                    scd_list.append({
                        '물품고유번호': asset_id,
                        'Start_Date': prev_row['변경일자'],
                        'End_Date': curr_date - pd.Timedelta(days=1),
                        'Is_Current': False,
                        '운용상태': prev_row['(변경)운용상태'],
                        '사유': prev_row['사유'],
                        '관리자': prev_row['관리자명']
                    })
                
                prev_row = row
            
            # 마지막 이력 (현재 상태)
            if prev_row is not None:
                scd_list.append({
                    '물품고유번호': asset_id,
                    'Start_Date': prev_row['변경일자'],
                    'End_Date': CURRENT_STATUS_END_DATE, # 2099-12-31
                    'Is_Current': True,
                    '운용상태': prev_row['(변경)운용상태'],
                    '사유': prev_row['사유'],
                    '관리자': prev_row['관리자명']
                })
        
        df_scd = pd.DataFrame(scd_list)
        print(f"      - SCD 이력 변환 완료: {len(df_scd)}건")
        
    else:
        print("      ❌ 오류: 이력 데이터에 '변경일자' 컬럼이 없습니다.")
        cols_scd = ['물품고유번호', 'Start_Date', 'End_Date', 'Is_Current', '운용상태', '사유', '관리자']
        df_scd = pd.DataFrame(columns=cols_scd)

# SCD 저장
df_scd.to_csv(os.path.join(SAVE_DIR, 'asset_scd_history.csv'), index=False, encoding='utf-8-sig')
# ---------------------------------------------------------
# 2. 데이터 정합성 검증 (Validation)
# ---------------------------------------------------------
print("\n🔍 [Phase 3] 데이터 정합성 검증 시작")

# 검증 1: 이력 기반 데이터 검증
current_snapshot = view_inventory_scd[
    view_inventory_scd['유효종료일자'] == CURRENT_STATUS_END_DATE.strftime('%Y-%m-%d')
]
total_op = len(df_op)
current_snapshot_qty = pd.to_numeric(current_snapshot['수량'], errors='coerce').sum()

print(f"1. 최신 상태 동기화 검증: 운용대장({total_op}) vs 이력스냅샷({int(current_snapshot_qty)})")
if total_op == current_snapshot_qty:
    print("   ✅ PASS: 일치합니다.")
else:
    print("   ❌ FAIL: 데이터 불일치 발생.")

# 검증 2: 날짜 논리 확인
if not df_du.empty:
    # df_du에 취득일자가 없을 수도 있으므로 df_master_info와 병합된 view_du_item 사용 권장
    df_check = view_du_item.copy()
    df_check['취득일자'] = pd.to_datetime(df_check['취득일자'], errors='coerce')
    df_check['불용일자'] = pd.to_datetime(df_check['불용일자'], errors='coerce')

    # NaT 여부 집계
    invalid_mask = df_check['취득일자'].isna() | df_check['불용일자'].isna()
    invalid_cnt = invalid_mask.sum()

    # 유효 날짜만 비교
    valid_df = df_check[~invalid_mask]
    error_cnt = (valid_df['불용일자'] < valid_df['취득일자']).sum()

    print("2. 날짜 논리 검증 (취득일자 < 불용일자)")
    if error_cnt == 0:
        print("   ✅ PASS: 시간 순서 정상.")
    else:
        print(f"   ❌ FAIL: {error_cnt}건 시간 역전.")

    if invalid_cnt > 0:
        print(f"   ⚠️ 참고: 날짜 누락/형식 오류 {invalid_cnt}건 존재.")
else:
    print("   ℹ️ 불용 데이터가 없어 검증 건너뜀.")

# [Fix] 처분 상태 동기화 검증 (승인 상태 고려)
if not df_dp.empty:
    # 처분 목록 중 '확정'인 건들만 추출
    confirmed_disposal_ids = df_dp[df_dp['승인상태'] == '확정']['물품고유번호'].unique()
    
    # 운용대장에서 해당 ID 조회
    if len(confirmed_disposal_ids) == 0:
        # 확정된 처분 건이 없는 경우
        print("3. 처분 상태(확정건): ℹ️ 확정된 처분 건이 없어 검증 건너뜀.")
    
    # 상태가 '처분'이 아닌 것 카운트
    else:
        op_status = df_op[df_op['물품고유번호'].isin(confirmed_disposal_ids)]['운용상태']
        err_cnt = (op_status != '처분').sum()
        print(f"3. 처분 상태(확정건): {'✅ PASS' if err_cnt == 0 else f'❌ FAIL ({err_cnt}건 미반영)'}")
    
    # 대기/반려 상태인 건수 출력 (ID 기준이 아닌 Row 기준 집계)
    # 기존: len(df_dp) - len(confirmed_ids) -> 중복 ID나 로직 오류 가능성 있음
    pending_cnt = (df_dp['승인상태'] != '확정').sum()
    print(f"   (참고) 진행 중인 처분 건: {pending_cnt}건")

print("\n🎉 모든 작업이 완료되었습니다.")