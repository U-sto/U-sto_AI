import pandas as pd
import numpy as np
import os
from pandas.errors import EmptyDataError

# ---------------------------------------------------------
# 0. 설정 및 데이터 로드
# ---------------------------------------------------------
# [Professor Fix 1] 기준일 고정
FIXED_TODAY_STR = "2026-02-10"
today = pd.to_datetime(FIXED_TODAY_STR).normalize()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.join(BASE_DIR, "data_lifecycle")
SAVE_DIR = os.path.join(BASE_DIR, "data_ml")
os.makedirs(SAVE_DIR, exist_ok=True)

print("📂 [Phase 4] AI 학습용 데이터 전처리 시작...")

# [Copilot Fix] 병합 시 필요한 컬럼 정의 (파일 누락 시 KeyError 방지용)
COLS_RT = ['물품고유번호', '반납일자', '반납확정일자', '사유', '승인상태']
COLS_DU = ['물품고유번호', '불용일자', '불용확정일자', '사유', '승인상태']
COLS_DP = ['물품고유번호', '처분방식', '처분확정일자', '물품상태', '승인상태']

# [Copilot Fix] 안전한 파일 로딩 함수 정의 (expected_cols 추가)
def load_csv_safe(filename, required=False, expected_cols=None):
    filepath = os.path.join(LOAD_DIR, filename)
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except EmptyDataError:
            # 파일은 있지만 비어있는 경우
            if expected_cols:
                return pd.DataFrame(columns=expected_cols)
            return pd.DataFrame()
    else:
        if required:
            print(f"❌ 필수 데이터 파일 누락: {filename}")
            exit()
        else:
            print(f"   ⚠️ 파일 없음 (빈 DataFrame 생성): {filename}")
            if expected_cols:
                return pd.DataFrame(columns=expected_cols)
            return pd.DataFrame()

# 1. 데이터 로드 (모든 생애주기 데이터)
df_op = load_csv_safe('04_01_operation_master.csv', required=True) # 운용 (필수)
df_rt = load_csv_safe('04_03_return_list.csv', expected_cols=COLS_RT)      # 반납
df_du = load_csv_safe('05_01_disuse_list.csv', expected_cols=COLS_DU)      # 불용
df_dp = load_csv_safe('06_01_disposal_list.csv', expected_cols=COLS_DP)    # 처분

print(f"   - 원천 데이터 로드 완료: 운용 대장 {len(df_op)}건")

# ---------------------------------------------------------
# [Review Fix] 데이터 중복 제거 (Deduplication)
# ---------------------------------------------------------
# 재사용 루프로 인해 동일 ID에 대한 반납/불용 이력이 여러 건 존재할 수 있음.
# 병합 시 Cartesian Product 방지를 위해, 각 자산별 '가장 최신' 이력 1건만 남김.

print("   > 이력 데이터 중복 제거(최신 건 유지) 수행 중...")

# 1) 반납 이력 중복 제거
if not df_rt.empty:
    # 날짜 형변환 (정렬을 위해)
    df_rt['반납일자'] = pd.to_datetime(df_rt['반납일자'], errors='coerce')
    df_rt['반납확정일자'] = pd.to_datetime(df_rt['반납확정일자'], errors='coerce')
    
    # 확정일 우선, 없으면 신청일 기준 내림차순 정렬 (최신 날짜가 위로)
    df_rt = df_rt.sort_values(
        by=['물품고유번호', '반납확정일자', '반납일자'], 
        ascending=[True, False, False],
        kind='mergesort'
    )
    # ID별 첫 번째 행만 유지
    df_rt = df_rt.drop_duplicates(subset=['물품고유번호'], keep='first')

# 2) 불용 이력 중복 제거
if not df_du.empty:
    df_du['불용일자'] = pd.to_datetime(df_du['불용일자'], errors='coerce')
    df_du['불용확정일자'] = pd.to_datetime(df_du['불용확정일자'], errors='coerce')
    
    df_du = df_du.sort_values(
        by=['물품고유번호', '불용확정일자', '불용일자'], 
        ascending=[True, False, False],
        kind='mergesort'
    )
    df_du = df_du.drop_duplicates(subset=['물품고유번호'], keep='first')

# 3) 처분 이력 중복 제거
if not df_dp.empty:
    df_dp['처분확정일자'] = pd.to_datetime(df_dp['처분확정일자'], errors='coerce')
    
    df_dp = df_dp.sort_values(
        by=['물품고유번호', '처분확정일자'], 
        ascending=[True, False],
        kind='mergesort'
    )
    df_dp = df_dp.drop_duplicates(subset=['물품고유번호'], keep='first')
# ---------------------------------------------------------
# 1. 데이터 병합 (Master Table 생성)
# ---------------------------------------------------------
print("   1. 생애주기 병합 (운용+반납+불용+처분)...")

# (1) 운용 + 반납 (Left Join)
# [Copilot Fix] 확정일자/승인상태 포함 및 컬럼명 충돌 방지
df_rt_subset = df_rt[['물품고유번호', '반납일자', '반납확정일자', '사유', '승인상태']].rename(
    columns={'승인상태': '반납승인상태'}
)
df_merged = pd.merge(df_op, df_rt_subset, on='물품고유번호', how='left')
df_merged.rename(columns={'사유': '상태변화'}, inplace=True) 

# (2) + 불용 (Left Join)
df_du_subset = df_du[['물품고유번호', '불용일자', '불용확정일자', '사유', '승인상태']].rename(
    columns={'사유': '불용사유', '승인상태': '불용승인상태'}
)
df_merged = pd.merge(df_merged, df_du_subset, on='물품고유번호', how='left')

# (3) + 처분 (Left Join)
df_dp_subset = df_dp[['물품고유번호', '처분방식', '처분확정일자', '물품상태', '승인상태']].rename(
    columns={'승인상태': '처분승인상태'}
)
df_merged = pd.merge(df_merged, df_dp_subset, on='물품고유번호', how='left')
# ---------------------------------------------------------
# 2. 전처리 및 결측치 보정 (Imputation)
# ---------------------------------------------------------
print("   2. 결측치 보정 및 기본 필드 정리...")

# [수정 2] 날짜 처리, 최종종료일 및 기준일 설정
# 1. 현재 시점 정의 (Today)
now = today # 코드 내 now 변수 호환용

# 2. 기본 날짜 컬럼 형변환
date_cols = ['취득일자', '반납일자', '불용일자']
for col in date_cols:
    if col in df_merged.columns:
        df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')

# 3. 확정일자 형변환 (존재하는 경우만)
confirm_cols = ['반납확정일자', '불용확정일자', '처분확정일자']
for col in confirm_cols:
    if col in df_merged.columns:
        df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')

# 4. 확정일자 우선순위 로직 적용
# (컬럼이 없으면 NaT Series 생성하여 에러 방지)
_ret_conf = df_merged['반납확정일자'] if '반납확정일자' in df_merged.columns else pd.Series(pd.NaT, index=df_merged.index)
_disuse_conf = df_merged['불용확정일자'] if '불용확정일자' in df_merged.columns else pd.Series(pd.NaT, index=df_merged.index)
_disp_conf = df_merged['처분확정일자'] if '처분확정일자' in df_merged.columns else pd.Series(pd.NaT, index=df_merged.index)

# 우선순위: 처분확정 > 불용확정 > 반납확정 (가장 늦은 단계의 확정일이 실제 종료 시점)
confirmed_end_date = _disp_conf.combine_first(_disuse_conf).combine_first(_ret_conf)

# 4. 차선책: 신청일자 (승인상태 '확정'인 경우만)
# 반납일자 유효성 체크
valid_ret_date = df_merged['반납일자'].where(df_merged['반납승인상태'] == '확정')
# 불용일자 유효성 체크
valid_disuse_date = df_merged['불용일자'].where(df_merged['불용승인상태'] == '확정')
# Fallback 구성: 확정된 반납일자 우선, 없으면 확정된 불용일자
fallback_end_date = valid_ret_date.combine_first(valid_disuse_date)

# 최종 종료일 도출: 확정일자 우선 적용, 없으면 신청일자 적용
df_merged['최종종료일'] = confirmed_end_date.combine_first(fallback_end_date)

# 5. [중요] '기준일' 컬럼 생성 (종료일이 없으면 오늘 날짜 기준)
# 이 컬럼이 생성되어야 뒤쪽의 파생변수(운용연차 등) 계산이 가능합니다.
df_merged['기준일'] = df_merged['최종종료일'].fillna(today)

# [DataFrame 초기화] 정의서 순서대로 데이터 구성
df_final = pd.DataFrame()

# --- A. 정적 정보 매핑 ---
df_final['물품고유번호'] = df_merged['물품고유번호']
df_final['G2B목록명'] = df_merged['G2B_목록명']
df_final['물품분류명'] = df_merged.get('물품분류명', df_merged['G2B_목록명'])
df_final['내용연수'] = df_merged['내용연수']
df_final['취득금액'] = df_merged['취득금액']
df_final['운용부서코드'] = df_merged['운용부서코드']
df_final['취득일자'] = df_merged['취득일자']
df_final['반납일자'] = df_merged['반납일자']
df_final['불용일자'] = df_merged['불용일자']
df_final['상태변화'] = df_merged['상태변화']
df_final['불용사유'] = df_merged['불용사유']
df_final['물품상태'] = df_merged['물품상태']
df_final['처분방식'] = df_merged['처분방식']
# [Copilot Fix] 학습여부 판단을 위해 병합된 승인상태/확정일자 정보를 임시로 매핑
df_final['처분승인상태'] = df_merged['처분승인상태']
df_final['처분확정일자'] = df_merged['처분확정일자']
df_final['운용부서명'] = df_merged['운용부서']
df_final['캠퍼스'] = df_merged['캠퍼스']
df_final['기준일'] = df_merged['기준일'] # 계산용 임시 컬럼

# --- B. 결측치 처리 (Imputation) ---
# 1) 취득금액 결측/0원: 중앙값(Median) 대체
# 안전장치: 데이터가 아예 없거나 양수 금액이 없는 경우 대비
valid_prices = df_final[df_final['취득금액'] > 0]['취득금액']
if not valid_prices.empty:
    median_price = valid_prices.median()
else:
    median_price = 1000000 # Default fallback (100만원)

df_final['취득금액'] = df_final['취득금액'].fillna(median_price).replace(0, median_price)

# 2) 내용연수 결측: 최빈값(Mode) 대체
# 안전장치: mode()가 비어있을 경우 대비
if not df_final['내용연수'].dropna().empty:
    mode_life = df_final['내용연수'].mode()[0]
    df_final['내용연수'] = df_final['내용연수'].fillna(mode_life)
else:
    df_final['내용연수'] = df_final['내용연수'].fillna(5) # Default fallback (5년)

# 3) 핵심 날짜(취득일자) NaT: 삭제 (생애주기 계산 불가)
initial_len = len(df_final)
df_final = df_final.dropna(subset=['취득일자'])
if initial_len != len(df_final):
    print(f"    - 결측치 처리: 취득일자 누락 {initial_len - len(df_final)}건 삭제됨")

# ---------------------------------------------------------
# 3. 파생 변수 생성 (Feature Engineering) - [보정된 값 사용]
# ---------------------------------------------------------
print("   3. 파생변수 생성 (보정된 데이터 기반)...")

# [Fix Error] 날짜 연산 전, 명시적으로 datetime64 타입으로 변환 (형식 통일)
df_final['기준일'] = pd.to_datetime(df_final['기준일'])
df_final['취득일자'] = pd.to_datetime(df_final['취득일자'])

# (1) 운용연차 (Years Used) & 운용월수
days_diff = (df_final['기준일'] - df_final['취득일자']).dt.days
# 음수 일수(미래 취득일자/기준일 역전 등) 보정: 0 미만은 0으로 clip
days_diff_clipped = days_diff.clip(lower=0)
df_final['운용연차'] = (days_diff_clipped / 365.0).round(2)
# 운용연차는 음수 방지를 위해 0 미만을 0으로 보정 (clip과 로직 일관성 유지)
df_final['운용연차'] = df_final['운용연차'].apply(lambda x: x if x > 0 else 0.0)
df_final['운용월수'] = (days_diff_clipped / 30.0).fillna(0).astype(int)

# (2) 취득월 (계절성)
df_final['취득월'] = df_final['취득일자'].dt.month

# (3) 학습데이터여부 산정 [Review Fix]
# 리뷰 반영: 매각을 포함하되, 수명 종료로 간주할 수 있는 명확한 조건 적용
# - 확실한 수명 종료: 폐기, 멸실
cond_disposal = df_final['처분방식'].isin(['폐기', '멸실'])

# - 매각 중 수명 종료로 볼 사유 (Phase 2에서 생성된 물리적/노후화 사유들)
#   제외 대상: '활용부서부재', '잉여물품' 등 (조기 매각 가능성 있음)
eol_reasons = ['고장/파손', '노후화(성능저하)', '수리비용과다', '구형화', '내구연한 경과(노후화)']
cond_sale_eol = (df_final['처분방식'] == '매각') & (df_final['불용사유'].isin(eol_reasons))

# 학습 대상 정의: 처분이 확정되었고, 기계적 수명이 다한 것(폐기/멸실 OR 노후화 매각)
is_disposal_confirmed = (df_final['처분승인상태'] == '확정') | df_final['처분확정일자'].notna()
is_mech_end = cond_disposal | cond_sale_eol
df_final['학습데이터여부'] = np.where(is_mech_end & is_disposal_confirmed, 'Y', 'N')

# 학습여부 판단 후 임시 컬럼 제거 (선택 사항, 저장 시 제외해도 됨)
df_final.drop(columns=['처분승인상태', '처분확정일자'], inplace=True)

# [Professor Fix 2] Feature Leakage 주의
# '잔여내용연수'는 (내용연수 - 운용연차)로 단순 계산되므로, 
# 내용연수가 법적 기준일 경우 모델이 실제 고장 패턴이 아니라 법적 기준만 학습할 위험이 있음.
# 따라서 '잔여내용연수'는 시각화용으로만 남기고, 학습 데이터(output_cols)에서는 제외하는 것을 권장.

# (4) 잔여내용연수 (학습 Feature에서는 제외할 것이지만 분석용으로 남겨둠)
df_final['잔여내용연수'] = (df_final['내용연수'] - df_final['운용연차']).round(2)

# (5) 부서가혹도 (Department Severity)
def get_severity(dept_name):
    if pd.isna(dept_name): return 1.0
    dept_str = str(dept_name)
    # 고부하 부서
    if any(k in dept_str for k in ['소프트웨어', '공학', '전산', 'AI', '정보','공과', '컴퓨터']):
        return 1.3
    # 중부하 부서
    if any(k in dept_str for k in ['연구', '실험', '과학']):
        return 1.2
    return 1.0

df_final['부서가혹도'] = df_final['운용부서명'].apply(get_severity)

# (6) 누적사용부하 (운용연차가 포함되므로 학습 Feature 제외 후보)
df_final['누적사용부하'] = (df_final['운용연차'] * df_final['부서가혹도']).round(2)

# (7) 고장임박도 (학습 Feature 제외 후보)
ratio = df_final['운용연차'] / df_final['내용연수']
df_final['고장임박도'] = (ratio ** 2).clip(0, 1).round(2)

# (8) 가격민감도 - [보정된 취득금액 사용]
log_price = np.log1p(df_final['취득금액'])
max_log_price = np.log1p(100000000) 
df_final['가격민감도'] = (log_price / max_log_price).clip(0, 1).round(2)

# (9) 리드타임등급 - [보정된 취득금액 사용]
def get_lead_time_grade(price):
    if pd.isna(price): return 1 # Default
    if price < 5000000: return 0
    elif price < 30000000: return 1
    else: return 2
df_final['리드타임등급'] = df_final['취득금액'].apply(get_lead_time_grade)

# (10) 장비중요도 - [계산된 민감도, 리드타임등급 사용]
df_final['장비중요도'] = ((df_final['가격민감도'] * 0.7) + ((df_final['리드타임등급'] * 0.5) * 0.3)).round(2)

# ---------------------------------------------------------
# [Phase 4-1] 추가 피처 엔지니어링 및 인코딩
# ---------------------------------------------------------
print("   > [4-1] 타겟 레이블링 및 범주형 데이터 수치화 수행...")

# (11) 타겟 데이터(Y) 생성: '실제수명' (Total Lifespan)
# 학습용 데이터(Y)인 경우, 이미 수명이 끝났으므로 '운용연차'가 곧 '실제수명'이 됨
# 예측용 데이터(N)인 경우, 아직 수명을 모르므로 NaN 처리
# [Target Definition]
# Regression Target: '실제수명' (Total Lifespan)
# 예측 시: 모델이 예측한 '예측_실제수명' - '현재_운용연차' = '예측_잔여수명(RUL)'

# (11) 타겟 데이터(Y) 생성
# 학습용 데이터(Y): 운용연차 = 실제수명
# 예측용 데이터(N): 알 수 없음(NaN)
df_final['실제수명'] = np.nan
mask_train = df_final['학습데이터여부'] == 'Y'
df_final.loc[mask_train, '실제수명'] = df_final.loc[mask_train, '운용연차']

# (12) 범주형 데이터 수치화 (Label Encoding)
# 모델 학습을 위해 텍스트(String) 데이터를 숫자(Code)로 변환
# ⚠️ Data Leakage 방지를 위해 '처분방식', '상태변화'는 인코딩 대상 및 학습 Feature에서 제외
categorical_cols = ['G2B목록명', '물품분류명', '운용부서코드', '캠퍼스',]

for col in categorical_cols:
    # 결측치는 'Unknown'으로 채운 후 인코딩 (안전장치)
    df_final[col] = df_final[col].fillna('Unknown').astype(str)
    
    # pd.factorize 사용 (sort=True를 해야 알파벳 순으로 번호가 매겨져 재현성 유지됨)
    # codes: 숫자로 변환된 배열, uniques: 고유값 리스트
    codes, uniques = pd.factorize(df_final[col], sort=True)
    
    # 원본 컬럼은 유지하고, '_Code' 붙은 수치화 컬럼 생성
    df_final[f'{col}_Code'] = codes

    # (옵션) 인코딩 매핑 정보 출력 (확인용)
    print(f"     - {col} 매핑 완료: {len(uniques)}개 항목")

# --- C. 예측값/결과값 (Placeholder) ---
df_final['실제잔여수명'] = np.nan 
df_final['예측잔여수명'] = np.nan
df_final['(월별)고장예상수량'] = 0
df_final['안전재고'] = 0
df_final['필요수량'] = 0
df_final['AI예측고장일'] = pd.NaT
df_final['안전버퍼'] = 0.0
df_final['권장발주일'] = pd.NaT
df_final['예측실행일자'] = today.strftime('%Y-%m-%d')
# 실제잔여수명: 학습 데이터는 0(이미 종료됨), 예측 데이터는 미지수
df_final.loc[df_final['학습데이터여부'] == 'Y', '실제잔여수명'] = 0.0

# ---------------------------------------------------------
# 4. 이상치 제거 (Outlier Removal)
# ---------------------------------------------------------
print("   4. 이상치 제거 수행...")
before_cnt = len(df_final)

#  논리적 이상치: 운용연차가 음수인 경우
df_final = df_final[df_final['운용연차'] >= 0]

print(f"    - 이상치 제거: {before_cnt - len(df_final)}건 제거됨")

# ---------------------------------------------------------
# 5. 데이터 분할 (Time Series Split) 및 저장
# ---------------------------------------------------------
print("   5. 시계열 기준 데이터 분할 (Train/Valid/Test)...")

# 학습용 데이터(Y)만 분할 대상
df_train_source = df_final[df_final['학습데이터여부'] == 'Y'].copy()
df_pred_source = df_final[df_final['학습데이터여부'] == 'N'].copy() 

# 시간 순 정렬
df_train_source = df_train_source.sort_values(by='취득일자')

# 분할 인덱스 계산
n_total = len(df_train_source)
n_train = int(n_total * 0.7)
n_valid = int(n_total * 0.2)
n_test = n_total - n_train - n_valid

# 데이터 슬라이싱
train_set = df_train_source.iloc[:n_train]
valid_set = df_train_source.iloc[n_train : n_train + n_valid]
test_set  = df_train_source.iloc[n_train + n_valid :]

# '데이터세트구분' 컬럼 추가
df_final['데이터세트구분'] = 'Prediction'
df_final.loc[train_set.index, '데이터세트구분'] = 'Train'
df_final.loc[valid_set.index, '데이터세트구분'] = 'Valid'
df_final.loc[test_set.index,  '데이터세트구분'] = 'Test'

print(f"        [Split 결과]")
print(f"   - Train (70%) : {len(train_set)}건")
print(f"   - Valid (20%) : {len(valid_set)}건")
print(f"   - Test  (10%) : {len(test_set)}건")
print(f"   - Pred  (운용) : {len(df_pred_source)}건")

# 최종 저장
# [Professor Fix 2] 최종 저장 컬럼 리스트 업데이트 (Leakage 변수 제외)
output_cols = [
    # 식별 정보
    '물품고유번호', 'G2B목록명', '물품분류명', '운용부서명', '캠퍼스',
    
    # 원본 메타데이터 (참고용)
    '취득일자', '반납일자', '불용일자', '처분방식', '물품상태', '불용사유',
    '상태변화', # <-- 참고용으로는 남기되 모델 입력(Code)에서는 제외
    
    # [입력 Feature] - 시점/상태 독립적 속성
    '내용연수', '취득금액', '부서가혹도', 
    '가격민감도', '장비중요도', '리드타임등급', '취득월',
    'G2B목록명_Code', '물품분류명_Code', '운용부서코드_Code', '캠퍼스_Code',
    
    # [Target]
    '실제수명',
    
    # 데이터 구분
    '학습데이터여부', '데이터세트구분',
    
    # 결과 Placeholder
    '실제잔여수명', '예측잔여수명', '(월별)고장예상수량', '안전재고', '필요수량', 
    'AI예측고장일', '안전버퍼', '권장발주일', '예측실행일자',
    
    # 참고용 파생변수 (학습엔 안쓰지만 분석엔 필요)
    '운용연차', '운용월수', '잔여내용연수', '누적사용부하', '고장임박도'
]

# 컬럼 스키마 고정: 누락 컬럼은 NaN으로 생성 후 저장
missing_cols = [c for c in output_cols if c not in df_final.columns]
if missing_cols:
    print("[경고] 최종 출력 스키마에서 누락된 컬럼이 있어 NaN으로 채웁니다:")
    print("   - " + ", ".join(missing_cols))
    for c in missing_cols:
        df_final[c] = np.nan
df_export = df_final.reindex(columns=output_cols)
save_path = os.path.join(SAVE_DIR, 'phase4_training_data.csv')
df_export.to_csv(save_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"✅ 처분 완료(학습용) 데이터: {len(df_export[df_export['학습데이터여부']=='Y'])} 건")
print(f"✅ 운용 중(예측용) 데이터 : {len(df_export[df_export['학습데이터여부']=='N'])} 건")
print(f"💾 최종 파일 저장 완료: {save_path}")
print("-" * 50)