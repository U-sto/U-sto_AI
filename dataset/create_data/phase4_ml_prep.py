import pandas as pd
import numpy as np
import os
from pandas.errors import EmptyDataError
# ---------------------------------------------------------
# 0. 설정 및 데이터 로드
# ---------------------------------------------------------
# 기준일을 시스템 '오늘'로 잡으면 코드를 돌릴 때마다 수명과 잔여일수가 달라져서 
# 모델 재현성(Reproducibility)이 떨어지므로 특정 일자로 고정하는 것이 좋음
FIXED_TODAY_STR = "2026-02-10"
today = pd.to_datetime(FIXED_TODAY_STR).normalize()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.join(BASE_DIR, "data_lifecycle")
SAVE_DIR = os.path.join(BASE_DIR, "data_ml")
os.makedirs(SAVE_DIR, exist_ok=True)

print("📂 [Phase 4] AI 학습용 데이터 전처리 시작...")

# 빈 데이터프레임이 로드될 경우를 대비해, 병합 시 KeyError가 나지 않도록 필수 컬럼을 명시
COLS_DU = ['물품고유번호', '불용일자', '불용확정일자', '사유', '승인상태']
COLS_DP = ['물품고유번호', '처분방식', '처분확정일자', '물품상태', '승인상태']

def load_csv_safe(filename, required=False, expected_cols=None):
    """안전하게 CSV 파일을 불러오고, 없거나 비어있으면 빈 DataFrame을 반환하는 함수"""
    filepath = os.path.join(LOAD_DIR, filename)
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except EmptyDataError:
            return pd.DataFrame(columns=expected_cols) if expected_cols else pd.DataFrame()
    else:
        if required:
            print(f"❌ 필수 데이터 파일 누락: {filename}")
            exit()
        return pd.DataFrame(columns=expected_cols) if expected_cols else pd.DataFrame()

# 1. 원천 데이터 로드
df_op = load_csv_safe('04_01_operation_master.csv', required=True)    
df_du = load_csv_safe('05_01_disuse_list.csv', expected_cols=COLS_DU)      
df_dp = load_csv_safe('06_01_disposal_list.csv', expected_cols=COLS_DP)    

print(f"   - 원천 데이터 로드 완료: 운용 대장 {len(df_op)}건")

# 하나의 물품이 여러 번 불용 이력을 가질 수 있으므로
# 최신 이력(확정일자 기준 내림차순) 하나만 남기고 중복을 제거해야 1:1 병합이 깔끔하게 됨
def drop_duplicates_safe(df, date_col, conf_date_col):
    if not df.empty:
        # 원본 DataFrame이 함수 호출로 인해 예상치 못하게 변경되지 않도록 복사본에서 작업
        df = df.copy()
        # 기준일자 컬럼을 datetime으로 변환
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if conf_date_col in df.columns:
            # 확정일자 컬럼이 존재하면 이를 기준으로도 정렬
            df[conf_date_col] = pd.to_datetime(df[conf_date_col], errors='coerce')
            df = df.sort_values(by=['물품고유번호', conf_date_col, date_col], ascending=[True, False, False], kind='mergesort')
        else:
            # 확정일자가 없더라도 최소한 물품고유번호+기준일자 기준으로 최신 이력을 선택
            df = df.sort_values(by=['물품고유번호', date_col], ascending=[True, False], kind='mergesort')
        return df.drop_duplicates(subset=['물품고유번호'], keep='first')
    return df

df_du = drop_duplicates_safe(df_du, '불용일자', '불용확정일자')
df_dp = drop_duplicates_safe(df_dp, '처분확정일자', '처분확정일자') # 처분은 일자가 하나뿐이므로 동일하게 처리

# ---------------------------------------------------------
# 1. 데이터 병합 (Master Table 생성)
# ---------------------------------------------------------
print("   1. 생애주기 병합 (운용+불용+처분)...")

# [Review 반영] 병합 전 컬럼 존재 여부 확인 함수
def get_existing_cols(df, target_cols):
    return [c for c in target_cols if c in df.columns]

# 운용 마스터에 불용, 처분 이력을 Left Join으로 붙임
# (1) 불용 이력 병합
df_du_sub = df_du[['물품고유번호', '불용일자', '불용확정일자', '사유', '승인상태']].copy()
df_du_sub = df_du_sub.rename(columns={'사유': '불용사유', '승인상태': '불용승인상태'})
df_merged = pd.merge(df_op, df_du_sub, on='물품고유번호', how='left')

# (2) 처분 이력 병합
df_dp_sub = df_dp[['물품고유번호', '처분방식', '처분확정일자', '물품상태', '승인상태']].copy()
df_dp_sub = df_dp_sub.rename(columns={'승인상태': '처분승인상태'})
df_merged = pd.merge(df_merged, df_dp_sub, on='물품고유번호', how='left')
# ---------------------------------------------------------
# 2. 전처리 및 결측치 보정 
# ---------------------------------------------------------
print("   2. 결측치 보정 및 기준일 산출...")

date_cols = ['취득일자', '불용일자', '불용확정일자', '처분확정일자']
for col in date_cols:
    if col in df_merged.columns:
        df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')

# 수명 계산의 끝점이 되는 '종료일'을 구하는 로직
# 1순위: 처분확정일자 > 2순위: 불용확정일자
confirmed_end_date = (
    df_merged.get('처분확정일자', pd.Series(index=df_merged.index, dtype='datetime64[ns]'))
    .combine_first(df_merged.get('불용확정일자', pd.Series(index=df_merged.index, dtype='datetime64[ns]')))
)

# 확정일자가 없을 경우 불용일자를 차선책으로 사용
valid_disuse_date = df_merged['불용일자'].where(df_merged.get('불용승인상태') == '확정')

df_merged['최종종료일'] = confirmed_end_date.combine_first(valid_disuse_date)
# 종료일이 없으면 '현재 운용 중'이라는 뜻이므로 기준일을 today로 설정
df_merged['기준일'] = df_merged['최종종료일'].fillna(today)

# 머신러닝용 DF 구성 (확정 상태 체크를 위해 관련 컬럼 일시 포함)
df_final = df_merged.copy()

# 안전한 컬럼 매핑 로직
df_final = pd.DataFrame(index=df_merged.index)
df_final['물품고유번호'] = df_merged['물품고유번호']
df_final['취득금액'] = df_merged.get('취득금액', 0)
df_final['운용부서코드'] = df_merged.get('운용부서코드')
df_final['캠퍼스'] = df_merged.get('캠퍼스')
df_final['취득일자'] = df_merged.get('취득일자')
df_final['불용일자'] = df_merged.get('불용일자')
df_final['불용사유'] = df_merged.get('불용사유')
df_final['물품상태'] = df_merged.get('물품상태')
df_final['처분방식'] = df_merged.get('처분방식')
df_final['기준일'] = df_merged.get('기준일')
# Fallback을 포함한 명칭 매핑
df_final['G2B목록명'] = df_merged['G2B_목록명'] if 'G2B_목록명' in df_merged.columns else df_merged.get('G2B목록명', pd.NA)
df_final['물품분류명'] = df_merged['물품분류명'] if '물품분류명' in df_merged.columns else df_final['G2B목록명']
df_final['운용부서명'] = df_merged['운용부서'] if '운용부서' in df_merged.columns else df_merged.get('운용부서명', pd.NA)
df_final['내용연수'] = df_merged['내용연수'] if '내용연수' in df_merged.columns else pd.Series(5, index=df_merged.index)

df_final['불용승인상태'] = df_merged.get('불용승인상태')              # 불용 승인상태 컬럼 보존
df_final['불용확정일자'] = df_merged.get('불용확정일자')              # 불용 확정일자 컬럼 보존
df_final['처분승인상태'] = df_merged.get('처분승인상태')              # 처분 승인상태 컬럼 보존
df_final['처분확정일자'] = df_merged.get('처분확정일자')  

# 결측치 보정 (가격이 0이거나 없는 경우 중앙값으로, 내용연수가 없으면 최빈값(보통 5년)으로)
valid_prices = df_final.loc[df_final['취득금액'] > 0, '취득금액']
median_price = valid_prices.median() if not valid_prices.empty else 1000000 # 기본값으로 100만원 사용
df_final['취득금액'] = df_final['취득금액'].fillna(median_price).replace(0, median_price)

df_final['내용연수'] = df_final['내용연수'].fillna(df_final['내용연수'].mode()[0] if not df_final['내용연수'].mode().empty else 5)
df_final = df_final.dropna(subset=['취득일자']) # 시작일이 없으면 수명 계산이 불가하므로 제거

# ---------------------------------------------------------
# 3. 파생 변수 생성 및 타겟 정의
# ---------------------------------------------------------
print("   3. 파생변수 생성 및 이상치 처리...")

# 운용연차 산출 (년 단위 환산)
df_final['운용연차'] = ((df_final['기준일'] - df_final['취득일자']).dt.days.clip(lower=0) / 365.0).round(2)

# 안전한 데이터 추출 함수
def safe_get_series(df, col_name, fill_val=np.nan):
    if col_name in df.columns:
        return df[col_name]
    return pd.Series(fill_val, index=df.index)

# [FIX] 에러가 났던 지점: .get() 대신 safe_get_series 사용
# 처분 또는 불용이 '확정'되었거나 날짜가 기록된 경우를 확정으로 간주
disposal_confirmed = (safe_get_series(df_final, '처분승인상태', '') == '확정') | (safe_get_series(df_final, '처분확정일자').notna())
disuse_confirmed = (safe_get_series(df_final, '불용승인상태', '') == '확정') | (safe_get_series(df_final, '불용확정일자').notna())

# 처분방식/불용사유 + 확정 여부를 함께 고려
is_disposal = disposal_confirmed & df_final['처분방식'].isin(['폐기', '멸실'])
is_sale_eol = (
    disuse_confirmed 
    & (df_final['처분방식'] == '매각') 
    & df_final['불용사유'].isin(['고장/파손', '노후화(성능저하)', '수리비용과다', '구형화', '내구연한 경과(노후화)'])
)
df_final['학습데이터여부'] = np.where(is_disposal | is_sale_eol, 'Y', 'N')

# --- IQR 기반 이상치(Outlier) 제거 ---
# 이유: 학습 데이터에 수명이 0.1년이거나 50년인 극단적 데이터가 섞여 있으면 모델이 흔들림.
train_cond = df_final['학습데이터여부'] == 'Y'
if train_cond.sum() > 0 and df_final.loc[train_cond, '운용연차'].notna().any():
    Q1 = df_final.loc[train_cond, '운용연차'].quantile(0.25)
    Q3 = df_final.loc[train_cond, '운용연차'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0* IQR
    upper_bound = Q3 + 2.0 * IQR

    # 이상치 제외 및 결과 출력
    outlier_mask = (df_final['학습데이터여부'] == 'Y') & ((df_final['운용연차'] < lower_bound) | (df_final['운용연차'] > upper_bound))
    df_outliers = df_final[outlier_mask]
    
    if not df_outliers.empty:
        print(f"      * [이상치 제거] 정상범위({lower_bound:.2f}~{upper_bound:.2f}년) 외 {len(df_outliers)}건 제외")

    valid_data_mask = (df_final['학습데이터여부'] == 'N') | ((df_final['학습데이터여부'] == 'Y') & (~outlier_mask))
    df_final = df_final[valid_data_mask].copy()

    # 이상치 데이터 통계 출력 (네가 원했던 부분!)
    if not df_outliers.empty:
        outlier_min = df_outliers['운용연차'].min()
        outlier_max = df_outliers['운용연차'].max()
        outlier_mode = df_outliers['운용연차'].mode()[0] if not df_outliers['운용연차'].mode().empty else '없음'
        
        print(f"      * [이상치 상세 분석] 제외 예정인 데이터 {len(df_outliers)}건의 수명 정보:")
        print(f"        - 정상 허용 범위: {lower_bound:.2f}년 ~ {upper_bound:.2f}년")
        print(f"        - 통계 ➔ 최소값: {outlier_min}년 / 최대값: {outlier_max}년 / 최빈값: {outlier_mode}년")
        
        # 어떤 물품들이 주로 걸렸는지 상위 5개 품목명 확인
        top_items = df_outliers['G2B목록명'].value_counts().head(3).to_dict()
        print(f"        - 주로 걸러진 품목 Top 3: {top_items}")

# 추가 파생 변수 산출
df_final['잔여내용연수'] = (df_final['내용연수'] - df_final['운용연차']).round(2)

def get_severity(dept_name):
    if pd.isna(dept_name): return 1.0
    dept_str = str(dept_name)
    if any(k in dept_str for k in ['소프트웨어', '공학', '전산', 'AI', '정보','공과', '컴퓨터']): return 1.3
    if any(k in dept_str for k in ['연구', '실험', '과학']): return 1.2
    return 1.0

df_final['부서가혹도'] = df_final['운용부서명'].apply(get_severity)
df_final['누적사용부하'] = (df_final['운용연차'] * df_final['부서가혹도']).round(2)
df_final['고장임박도'] = ((df_final['운용연차'] / df_final['내용연수'].replace(0, np.nan)) ** 2).clip(0, 1).round(2)

# 예산/구매 관련 지표
df_final['가격민감도'] = (np.log1p(df_final['취득금액']) / np.log1p(100000000)).clip(0, 1).round(2)
df_final['리드타임등급'] = df_final['취득금액'].apply(lambda x: 0 if x < 5000000 else (1 if x < 30000000 else 2))
df_final['장비중요도'] = ((df_final['가격민감도'] * 0.7) + ((df_final['리드타임등급'] * 0.5) * 0.3)).round(2)
df_final['취득월'] = df_final['취득일자'].dt.month

# ---------------------------------------------------------
# 4. 컬럼정의서 기반 결과 Placeholder 세팅 및 인코딩
# ---------------------------------------------------------
# 타겟 세팅 (실제수명: 학습에 쓰일 Y값)
df_final['실제수명'] = np.nan
df_final.loc[df_final['학습데이터여부'] == 'Y', '실제수명'] = df_final.loc[df_final['학습데이터여부'] == 'Y', '운용연차']

# 컬럼정의서 필수 산출물 빈칸(Placeholder) 처리
# (이 값들은 Phase 6 예측 단계나 LLM 후처리 단계에서 채워짐)
df_final['서비스계수'] = np.nan 
df_final['실제잔여수명'] = np.where(df_final['학습데이터여부'] == 'Y', 0.0, np.nan)
df_final['예측잔여수명'] = np.nan
df_final['(월별)고장예상수량'] = 0
df_final['안전재고'] = 0
df_final['(월별)필요수량'] = 0
df_final['AI예측고장일'] = pd.NaT
df_final['안전버퍼'] = 0.0
df_final['권장발주일'] = pd.NaT
df_final['예측실행일자'] = today.strftime('%Y-%m-%d')

# 범주형 수치화 (모델이 이해할 수 있도록 라벨 인코딩)
for col in ['G2B목록명', '물품분류명', '운용부서코드', '캠퍼스']:
    df_final[col] = df_final[col].fillna('Unknown').astype(str)
    df_final[f'{col}_Code'] = pd.factorize(df_final[col], sort=True)[0]

# ---------------------------------------------------------
# 5. 데이터 분할 및 저장 (Train / Valid / Test / Pred)
# ---------------------------------------------------------
df_train_source = df_final[df_final['학습데이터여부'] == 'Y'].copy()

# 시간 기반 분할: Train(70%) / Valid(20%) / Test(10%)
# - '취득일자'를 기준으로 시계열 블록 분할을 수행
# - 실제 운영에서는 과거 데이터로 미래를 예측하므로, 테스트 세트는 항상 가장 최근 데이터가 되도록 구성
date_col = '취득일자'
df_time_split = df_train_source.copy()

# 확실한 시계열 계산을 위해 datetime 타입으로 변환
df_time_split[date_col] = pd.to_datetime(df_time_split[date_col])

# 70%, 90% 분위수 기준으로 시점 경계 설정 (≈ Train 70% / Valid 20% / Test 10%)
quantiles = df_time_split[date_col].quantile([0.7, 0.9])
train_cutoff = quantiles.loc[0.7]
test_cutoff = quantiles.loc[0.9]

train_idx = df_time_split[df_time_split[date_col] < train_cutoff].index
valid_idx = df_time_split[
    (df_time_split[date_col] >= train_cutoff) & (df_time_split[date_col] < test_cutoff)
].index
test_idx = df_time_split[df_time_split[date_col] >= test_cutoff].index

# 데이터세트 구분 컬럼에 결과 매핑
df_final['데이터세트구분'] = 'Prediction'  # 기본값은 예측 대상 (학습데이터여부 == 'N')
df_final.loc[train_idx, '데이터세트구분'] = 'Train'
df_final.loc[valid_idx, '데이터세트구분'] = 'Valid'
df_final.loc[test_idx, '데이터세트구분'] = 'Test'

# 최종 출력 컬럼 지정 (컬럼정의서 매핑 반영 & 데이터 누수 방지)
output_cols = [
    # 정적 & 기본 정보
    '물품고유번호', 'G2B목록명', '물품분류명', '운용부서코드', '운용부서명', '캠퍼스',
    '취득일자', '불용일자', '처분방식', '물품상태', '불용사유', 
    
    # 파생 변수 (Features)
    '내용연수', '취득금액', '운용연차', '잔여내용연수', '부서가혹도', '누적사용부하',
    '고장임박도', '가격민감도', '장비중요도', '리드타임등급', '취득월',
    'G2B목록명_Code', '물품분류명_Code', '운용부서코드_Code', '캠퍼스_Code',
    
    # 타겟 및 구분
    '실제수명', '학습데이터여부', '데이터세트구분',
    
    # 예측값/결과값 (컬럼정의서 매핑 완벽 대응)
    '서비스계수', '실제잔여수명', '예측잔여수명', '(월별)고장예상수량', '안전재고', 
    '(월별)필요수량', 'AI예측고장일', '안전버퍼', '권장발주일', '예측실행일자'
]

# [Review 반영] reindex 전 컬럼 누락 확인
missing_cols = [c for c in output_cols if c not in df_final.columns]
if missing_cols:
    print(f"⚠️ 경고: 다음 컬럼이 생성되지 않았습니다: {missing_cols}")

df_export = df_final.reindex(columns=output_cols)
save_path = os.path.join(SAVE_DIR, 'phase4_training_data.csv')
df_export.to_csv(save_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"✅ 처분 완료(학습용) 데이터: {len(df_export[df_export['학습데이터여부']=='Y'])} 건 (Train/Valid/Test 분할 완료)")
print(f"✅ 운용 중(예측용) 데이터 : {len(df_export[df_export['학습데이터여부']=='N'])} 건")
print(f"💾 최종 파일 저장 완료: {save_path}")
print("-" * 50)