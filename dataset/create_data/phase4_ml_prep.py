import pandas as pd
import numpy as np
import os
from datetime import datetime
from pandas.errors import EmptyDataError

# ---------------------------------------------------------
# 0. 설정 및 데이터 로드
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.join(BASE_DIR, "data_lifecycle")
SAVE_DIR = os.path.join(BASE_DIR, "data_ml")
os.makedirs(SAVE_DIR, exist_ok=True)

print("📂 [Phase 4] AI 학습용 데이터 전처리 시작...")

# [Copilot Fix] 병합 시 필요한 컬럼 정의 (파일 누락 시 KeyError 방지용)
COLS_RT = ['물품고유번호', '반납일자', '사유']
COLS_DU = ['물품고유번호', '불용일자', '사유']
COLS_DP = ['물품고유번호', '처분방식', '물품상태', '승인상태']

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
# 1. 데이터 병합 (Master Table 생성)
# ---------------------------------------------------------
print("   1. 생애주기 병합 (운용+반납+불용+처분)...")

# (1) 운용 + 반납 (Left Join)
# df_rt가 빈 DF여도 컬럼이 있으므로 에러 없이 병합됨
df_merged = pd.merge(df_op, df_rt[['물품고유번호', '반납일자', '사유']], on='물품고유번호', how='left')
df_merged.rename(columns={'사유': '상태변화'}, inplace=True) 

# (2) + 불용 (Left Join)
df_du_subset = df_du[['물품고유번호', '불용일자', '사유']].rename(columns={'사유': '불용사유'})
df_merged = pd.merge(df_merged, df_du_subset, on='물품고유번호', how='left')

# (3) + 처분 (Left Join)
df_merged = pd.merge(df_merged, df_dp[['물품고유번호', '처분방식', '물품상태']], on='물품고유번호', how='left')

# ---------------------------------------------------------
# 2. 전처리 및 결측치 보정 (Imputation)
# ---------------------------------------------------------
print("   2. 결측치 보정 및 기본 필드 정리...")

# [수정 2] 날짜 처리, 최종종료일 및 기준일 설정
# 1. 현재 시점 정의 (Today)
today = pd.to_datetime(datetime.now().date())

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

# 차선책: 신청일자 (반납일자 > 불용일자)
fallback_end_date = df_merged['반납일자'].combine_first(df_merged['불용일자'])

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
df_final['운용부서명'] = df_merged['운용부서']
df_final['캠퍼스'] = df_merged['캠퍼스']
df_final['기준일'] = df_merged['기준일'] # 계산용 임시 컬럼

# --- B. 결측치 처리 (Imputation) ---
# [Copilot Fix] Feature Engineering 전에 결측치를 먼저 채움 + 안전장치 추가

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

# (1) 운용연차 (Years Used) & 운용월수
days_diff = (df_final['기준일'] - df_final['취득일자']).dt.days
df_final['운용연차'] = (days_diff / 365.0).round(2)
# 음수 값(미래 취득 등 오류) 보정
df_final['운용연차'] = df_final['운용연차'].apply(lambda x: x if x > 0 else 0.0)
df_final['운용월수'] = (days_diff / 30.0).fillna(0).astype(int)

# (2) 취득월 (계절성)
df_final['취득월'] = df_final['취득일자'].dt.month

# (3) 학습데이터여부
# 기계적 수명이 다한 것만 학습('Y'). 단순 매각이나 현재 운용 중인 것은 예측 대상('N')
is_mech_end = df_final['처분방식'].isin(['폐기', '멸실'])
df_final['학습데이터여부'] = np.where(is_mech_end, 'Y', 'N')

# (4) 잔여내용연수 (보정된 내용연수 사용)
df_final['잔여내용연수'] = (df_final['내용연수'] - df_final['운용연차']).round(2)

# (5) 부서가혹도 (Department Severity)
def get_severity(dept_name):
    if pd.isna(dept_name): return 1.0
    dept_str = str(dept_name)
    # 고부하 부서
    if any(k in dept_str for k in ['소프트웨어', '공학', '전산', 'AI', '정보', '컴퓨터']):
        return 1.3
    # 중부하 부서
    if '연구' in dept_str or '실험' in dept_str:
        return 1.2
    return 1.0

df_final['부서가혹도'] = df_final['운용부서명'].apply(get_severity)

# (6) 누적사용부하
df_final['누적사용부하'] = (df_final['운용연차'] * df_final['부서가혹도']).round(2)

# (7) 고장임박도 (Failure Imminence) - [보정된 내용연수 사용]
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

# ---------------------------------------------------------
# 4. 이상치 제거 (Outlier Removal)
# ---------------------------------------------------------
print("   4. 이상치 제거 수행...")
before_cnt = len(df_final)

# 1) 논리적 이상치: 운용연차가 음수인 경우
df_final = df_final[df_final['운용연차'] >= 0]

# 2) 통계적 이상치: 취득금액 상위 0.1% 제거 (왜곡 방지)
if not df_final.empty:
    q999 = df_final['취득금액'].quantile(0.999)
    df_final = df_final[df_final['취득금액'] <= q999]

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
output_cols = [
    '물품고유번호', 'G2B목록명', '물품분류명', '내용연수', '취득금액', '운용부서코드', 
    '취득일자', '반납일자', '불용일자', '상태변화', '불용사유', '물품상태', 
    '처분방식', '운용부서명', '캠퍼스',
    '운용연차', '학습데이터여부', '잔여내용연수', '부서가혹도', '누적사용부하', 
    '고장임박도', '가격민감도', '장비중요도', '리드타임등급',
    '실제잔여수명', '예측잔여수명', '(월별)고장예상수량', '안전재고', '필요수량', 
    'AI예측고장일', '안전버퍼', '권장발주일', '예측실행일자', '데이터세트구분', '운용월수', '취득월'
]

df_export = df_final.reindex(columns=output_cols)
save_path = os.path.join(SAVE_DIR, 'phase4_training_data.csv')
df_export.to_csv(save_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"✅ 처분 완료(학습용) 데이터: {len(df_export[df_export['학습데이터여부']=='Y'])} 건")
print(f"✅ 운용 중(예측용) 데이터 : {len(df_export[df_export['학습데이터여부']=='N'])} 건")
print(f"💾 최종 파일 저장 완료: {save_path}")
print("-" * 50)