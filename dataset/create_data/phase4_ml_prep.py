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
COLS_MT = [
    '점검수리일자', '물품고유번호', 'G2B_목록번호', 'G2B_목록명', '운용부서',
    '점검수리구분', '처리결과', '수리비용', '장애심각도', '비고'
]

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
df_mt = load_csv_safe('04_04_maintenance_list.csv', expected_cols=COLS_MT)

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
df_final['구매배치ID'] = df_merged.get('구매배치ID', '')
df_final['구매배치수량'] = df_merged.get('구매배치수량', 1)
df_final['대량구매여부'] = df_merged.get('대량구매여부', 0)
df_final['부서예산등급'] = df_merged.get('부서예산등급', 'MEDIUM')
df_final['부서교체성향'] = df_merged.get('부서교체성향', 1.0)
df_final['부서관리성향'] = df_merged.get('부서관리성향', 1.0)
df_final['월평균사용시간'] = df_merged.get('월평균사용시간', 0)
df_final['주당사용일수'] = df_merged.get('주당사용일수', 0)
df_final['공용장비여부'] = df_merged.get('공용장비여부', 0)
df_final['수업사용여부'] = df_merged.get('수업사용여부', 0)
df_final['사용강도지수'] = df_merged.get('사용강도지수', 1.0)

df_final['불용승인상태'] = df_merged.get('불용승인상태')              # 불용 승인상태 컬럼 보존
df_final['불용확정일자'] = df_merged.get('불용확정일자')              # 불용 확정일자 컬럼 보존
df_final['처분승인상태'] = df_merged.get('처분승인상태')              # 처분 승인상태 컬럼 보존
df_final['처분확정일자'] = df_merged.get('처분확정일자')  

# 결측치 보정 (가격이 0이거나 없는 경우 중앙값으로, 내용연수가 없으면 최빈값(보통 5년)으로)
valid_prices = df_final.loc[df_final['취득금액'] > 0, '취득금액']
median_price = valid_prices.median() if not valid_prices.empty else 1000000 # 기본값으로 100만원 사용
df_final['취득금액'] = df_final['취득금액'].fillna(median_price).replace(0, median_price)

df_final['내용연수'] = df_final['내용연수'].fillna(df_final['내용연수'].mode()[0] if not df_final['내용연수'].mode().empty else 5)
df_final['구매배치ID'] = df_final['구매배치ID'].fillna('').astype(str)
blank_batch_mask = df_final['구매배치ID'].str.strip() == ''
df_final.loc[blank_batch_mask, '구매배치ID'] = 'B' + df_final.loc[blank_batch_mask, '물품고유번호'].astype(str)
df_final['구매배치수량'] = pd.to_numeric(df_final['구매배치수량'], errors='coerce').fillna(1).astype(int)
df_final['대량구매여부'] = pd.to_numeric(df_final['대량구매여부'], errors='coerce').fillna(0).astype(int)
df_final['부서예산등급'] = df_final['부서예산등급'].fillna('MEDIUM').astype(str)
for col in ['부서교체성향', '부서관리성향', '월평균사용시간', '주당사용일수', '공용장비여부', '수업사용여부', '사용강도지수']:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
df_final['부서교체성향'] = df_final['부서교체성향'].fillna(1.0)
df_final['부서관리성향'] = df_final['부서관리성향'].fillna(1.0)
df_final['월평균사용시간'] = df_final['월평균사용시간'].fillna(0).astype(int)
df_final['주당사용일수'] = df_final['주당사용일수'].fillna(0).astype(int)
df_final['공용장비여부'] = df_final['공용장비여부'].fillna(0).astype(int)
df_final['수업사용여부'] = df_final['수업사용여부'].fillna(0).astype(int)
df_final['사용강도지수'] = df_final['사용강도지수'].fillna(1.0)
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
# 학습 타겟은 "물리적 수명 종료"에 가까운 건만 사용한다.
# 활용부서부재/구형화 같은 행정적 반납 사유를 섞으면 모델이 수명보다 조직 이동 패턴을 배우게 된다.
PHYSICAL_EOL_REASONS = [
    '고장/파손', '노후화', '노후화(성능저하)', '성능저하',
    '수리비용과다', '내용연수경과', '내구연한 경과(노후화)'
]
physical_eol_reason = df_final['불용사유'].isin(PHYSICAL_EOL_REASONS)
is_disposal = disposal_confirmed & df_final['처분방식'].isin(['폐기', '멸실']) & physical_eol_reason
is_sale_eol = (
    disuse_confirmed 
    & (df_final['처분방식'] == '매각') 
    & physical_eol_reason
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

# 구매 배치/코호트 feature: 결과 이후 정보가 아닌 동일 구매 건의 규모와 현재 운용연차 구조만 사용한다.
batch_stats = df_final.groupby('구매배치ID').agg(
    동일배치자산수=('물품고유번호', 'count'),
    동일배치운용연차평균=('운용연차', 'mean')
).reset_index()
df_final = df_final.merge(batch_stats, on='구매배치ID', how='left')
df_final['동일배치자산수'] = df_final['동일배치자산수'].fillna(1).astype(int)
df_final['동일배치운용연차평균'] = df_final['동일배치운용연차평균'].fillna(df_final['운용연차']).round(2)

def get_term(month):
    if month in [3, 4, 5, 6]:
        return '1학기'
    if month in [7, 8]:
        return '여름'
    if month in [9, 10, 11, 12]:
        return '2학기'
    return '겨울'

df_final['취득학기구분'] = df_final['취득월'].apply(get_term)
term_code_map = {'겨울': 0, '1학기': 1, '여름': 2, '2학기': 3}
budget_code_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
df_final['취득학기구분_Code'] = df_final['취득학기구분'].map(term_code_map).fillna(0).astype(int)
df_final['신학기준비월여부'] = df_final['취득월'].isin([2, 3, 8, 9]).astype(int)
df_final['예산집행월여부'] = (df_final['취득월'] == 12).astype(int)
df_final['방학기간여부'] = df_final['취득월'].isin([1, 2, 7, 8]).astype(int)
df_final['부서예산등급_Code'] = df_final['부서예산등급'].map(budget_code_map).fillna(1).astype(int)

# 점검/수리 이력 집계. 각 자산의 기준일 이전 이력만 사용해 미래 정보 누수를 막는다.
maintenance_features = pd.DataFrame(index=df_final.index)
maintenance_default_cols = [
    '누적점검수리횟수', '누적수리횟수', '최근1년수리횟수', '최근2년수리횟수',
    '마지막수리후경과개월', '누적수리비용', '취득금액대비수리비율',
    '최대장애심각도', '교체권고횟수'
]
for col in maintenance_default_cols:
    maintenance_features[col] = 0
maintenance_features['마지막수리후경과개월'] = 0.0
maintenance_features['취득금액대비수리비율'] = 0.0

if not df_mt.empty and {'물품고유번호', '점검수리일자'}.issubset(df_mt.columns):
    mt = df_mt.copy()
    mt['점검수리일자'] = pd.to_datetime(mt['점검수리일자'], errors='coerce')
    mt['수리비용'] = pd.to_numeric(mt.get('수리비용', 0), errors='coerce').fillna(0)
    mt['장애심각도'] = pd.to_numeric(mt.get('장애심각도', 0), errors='coerce').fillna(0)
    mt = mt.dropna(subset=['점검수리일자'])

    asset_base = df_final[['물품고유번호', '기준일', '취득금액']].copy()
    asset_base['row_index_for_merge'] = asset_base.index
    mt = mt.merge(asset_base, on='물품고유번호', how='inner')
    mt = mt[mt['점검수리일자'] <= mt['기준일']].copy()

    if not mt.empty:
        mt['is_repair'] = mt['점검수리구분'].isin(['수리', '부품교체', '장애점검']) | mt['처리결과'].isin(['경미수리', '주요수리', '교체권고'])
        mt['is_recent_1y'] = mt['점검수리일자'] >= (mt['기준일'] - pd.Timedelta(days=365))
        mt['is_recent_2y'] = mt['점검수리일자'] >= (mt['기준일'] - pd.Timedelta(days=730))
        mt['is_replace_recommend'] = mt['처리결과'] == '교체권고'

        agg = mt.groupby('row_index_for_merge').agg(
            누적점검수리횟수=('점검수리일자', 'count'),
            누적수리횟수=('is_repair', 'sum'),
            최근1년수리횟수=('is_recent_1y', 'sum'),
            최근2년수리횟수=('is_recent_2y', 'sum'),
            누적수리비용=('수리비용', 'sum'),
            최대장애심각도=('장애심각도', 'max'),
            교체권고횟수=('is_replace_recommend', 'sum'),
            마지막수리일자=('점검수리일자', 'max')
        )
        for col in ['누적점검수리횟수', '누적수리횟수', '최근1년수리횟수', '최근2년수리횟수', '누적수리비용', '최대장애심각도', '교체권고횟수']:
            maintenance_features.loc[agg.index, col] = agg[col]
        elapsed_months = ((df_final.loc[agg.index, '기준일'] - agg['마지막수리일자']).dt.days / 30.4375).clip(lower=0)
        maintenance_features.loc[agg.index, '마지막수리후경과개월'] = elapsed_months.round(1)

maintenance_features['마지막수리후경과개월'] = maintenance_features['마지막수리후경과개월'].where(
    maintenance_features['누적점검수리횟수'] > 0,
    (df_final['운용연차'] * 12).round(1)
)
maintenance_features['취득금액대비수리비율'] = (
    maintenance_features['누적수리비용'] / df_final['취득금액'].replace(0, np.nan)
).fillna(0).clip(0, 1).round(3)
df_final = pd.concat([df_final, maintenance_features], axis=1)

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

# ---------------------------------------------------------
# 5. 데이터 분할 및 타겟 인코딩 (Train / Valid / Test / Pred)
# ---------------------------------------------------------
def assign_itemwise_time_splits(df_source, group_col, date_col):
    """
    전체 날짜 분위수로 한 번에 자르면 특정 품목이 Test에 몰릴 수 있다.
    품목별로 실제 종료 이벤트 시점을 유지한 채 Train/Valid/Test를 나눠 평가 분포를 더 안정적으로 만든다.
    """
    train_idx, valid_idx, test_idx = [], [], []
    if df_source.empty:
        return train_idx, valid_idx, test_idx

    source = df_source.copy()
    source[date_col] = pd.to_datetime(source[date_col], errors='coerce')
    source = source.sort_values([group_col, date_col, '물품고유번호'], kind='mergesort')

    for _, group in source.groupby(group_col, sort=False):
        n = len(group)
        if n >= 10:
            n_train = max(1, int(np.floor(n * 0.70)))
            n_valid = max(1, int(np.floor(n * 0.20)))
            if n_train + n_valid >= n:
                n_valid = max(1, n - n_train - 1)
        elif n >= 5:
            n_train = max(2, int(np.floor(n * 0.70)))
            n_valid = 1
            if n_train + n_valid >= n:
                n_train = max(1, n - 2)
        elif n >= 3:
            n_train = n - 2
            n_valid = 1
        else:
            n_train = n
            n_valid = 0

        n_test = max(0, n - n_train - n_valid)
        train_idx.extend(group.index[:n_train])
        valid_idx.extend(group.index[n_train:n_train + n_valid])
        test_idx.extend(group.index[n_train + n_valid:n_train + n_valid + n_test])

    return train_idx, valid_idx, test_idx

df_train_source = df_final[df_final['학습데이터여부'] == 'Y'].copy()
train_idx, valid_idx, test_idx = assign_itemwise_time_splits(df_train_source, 'G2B목록명', '기준일')

# 데이터세트 구분 컬럼에 결과 매핑
df_final['데이터세트구분'] = 'Prediction'  # 기본값은 예측 대상 (학습데이터여부 == 'N')
df_final.loc[train_idx, '데이터세트구분'] = 'Train'
df_final.loc[valid_idx, '데이터세트구분'] = 'Valid'
df_final.loc[test_idx, '데이터세트구분'] = 'Test'

split_summary = df_final.loc[df_final['학습데이터여부'] == 'Y', '데이터세트구분'].value_counts()
print("✅ 품목별 시간 기반 Train/Valid/Test 분할 완료")
print(split_summary.to_string())

# 💡 [Upgrade] Target Encoding 적용 (기존 pd.factorize 대체)
# 타겟 인코딩 통계는 최종 Train split에서만 계산하여 Valid/Test 누수를 차단한다.
categorical_cols = ['G2B목록명', '물품분류명', '운용부서코드', '캠퍼스']
train_mask_te = df_final['데이터세트구분'] == 'Train'
global_mean_life = df_final.loc[train_mask_te, '실제수명'].mean()
if pd.isna(global_mean_life):
    global_mean_life = df_final.loc[df_final['학습데이터여부'] == 'Y', '실제수명'].mean()
if pd.isna(global_mean_life):
    global_mean_life = 5.0

for col in categorical_cols:
    df_final[col] = df_final[col].fillna('Unknown').astype(str)
    target_mean = df_final.loc[train_mask_te].groupby(col)['실제수명'].mean()
    category_counts = df_final.loc[train_mask_te, col].value_counts()
    smoothing_factor = 10
    smoothed_mean = (target_mean * category_counts + global_mean_life * smoothing_factor) / (category_counts + smoothing_factor)
    df_final[f'{col}_Code'] = df_final[col].map(smoothed_mean).fillna(global_mean_life)

print(f"✅ Train 기준 타겟 인코딩 완료 (결측치 대체값: {global_mean_life:.2f}년)")

# 최종 출력 컬럼 지정 (컬럼정의서 매핑 반영 & 데이터 누수 방지)
output_cols = [
    # 정적 & 기본 정보
    '물품고유번호', 'G2B목록명', '물품분류명', '운용부서코드', '운용부서명', '캠퍼스',
    '취득일자', '불용일자', '처분방식', '물품상태', '불용사유', 
    
    # 파생 변수 (Features)
    '내용연수', '취득금액', '운용연차', '잔여내용연수', '부서가혹도', '누적사용부하',
    '고장임박도', '가격민감도', '장비중요도', '리드타임등급', '취득월',
    '구매배치ID', '구매배치수량', '대량구매여부', '동일배치자산수', '동일배치운용연차평균',
    '월평균사용시간', '주당사용일수', '공용장비여부', '수업사용여부', '사용강도지수',
    '누적점검수리횟수', '누적수리횟수', '최근1년수리횟수', '최근2년수리횟수',
    '마지막수리후경과개월', '누적수리비용', '취득금액대비수리비율', '최대장애심각도', '교체권고횟수',
    '취득학기구분', '취득학기구분_Code', '신학기준비월여부', '예산집행월여부', '방학기간여부',
    '부서예산등급', '부서예산등급_Code', '부서교체성향', '부서관리성향',
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
