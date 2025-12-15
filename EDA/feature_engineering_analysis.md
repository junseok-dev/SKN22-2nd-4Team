# Feature Engineering Notebook 분석 보고서

## 1. 개요
이 문서는 `Feature_Engineering.ipynb` 노트북의 구조, 코드 흐름, 그리고 적용된 특징 공학(Feature Engineering) 기법들을 상세하게 분석한 보고서입니다. 해당 노트북은 머신러닝 모델의 성능을 향상시키기 위해 원본 데이터(고객 데이터 및 가격 데이터)를 가공하고 유의미한 파생 변수를 생성하는 과정을 담고 있습니다.

## 2. 라이브러리 및 데이터 로드
### 2.1 사용된 라이브러리
분석 및 데이터 처리를 위해 다음과 같은 주요 Python 라이브러리들이 사용되었습니다:
- **데이터 조작**: `pandas` (pd), `numpy` (np) - 대용량 데이터 처리 및 수치 연산.
- **시각화**: `seaborn` (sns), `matplotlib.pyplot` (plt) - 정적 그래프 및 통계 시각화.
- **인터랙티브 시각화**: `plotly.express` (px) - 대화형 그래프 생성.
- **날짜 처리**: `datetime` - 시계열 데이터 가공.

### 2.2 데이터셋 로딩
- `client_df`: 고객 관련 데이터 (`churn_data_modeling.csv`)를 로드.
- `price_df`: 가격 관련 데이터 (`price_data.csv`)를 로드.
- **전처리**: 날짜 관련 컬럼들(`date_activ`, `date_end`, `date_modif_prod`, `date_renewal`, `price_date`)을 `datetime` 객체로 변환하여 시계열 연산이 가능하도록 준비했습니다.

## 3. 주요 Feature Engineering 단계

### 3.1 가격 변동성 특징 생성 (Price Difference Features)
가격 데이터(`price_df`)를 활용하여 다양한 기간 및 시간대별 가격 차이를 계산함으로써, 가격 민감도가 이탈(Churn)에 미치는 영향을 포착하려 했습니다.

1.  **연말-연초 가격 차이 (Dec-Jan Difference)**:
    - 1월과 12월의 Off-peak(비피크) 시간대 에너지 및 전력 가격을 추출.
    - `offpeak_diff_dec_january_energy`: 12월과 1월의 에너지 가격 차이.
    - `offpeak_diff_dec_january_power`: 12월과 1월의 전력 가격 차이.
    - 이 변수들은 연말 가격 변동이 고객 이탈에 미치는 영향을 파악하기 위해 생성되었습니다.

2.  **기간별 평균 가격 차이 (Mean Difference between Periods)**:
    - 각 고객(`id`)별로 Off-peak, Peak, Mid-peak 등 시간대별 평균 가격을 계산.
    - `off_peak_peak_var_mean_diff`: Off-peak와 Peak 시간대의 변동 가격(energy) 평균 차이.
    - `peak_mid_peak_var_mean_diff`: Peak와 Mid-peak 시간대의 변동 가격 평균 차이.
    - `off_peak_mid_peak_var_mean_diff`: Off-peak와 Mid-peak 시간대의 변동 가격 평균 차이.
    - 고정 가격(power)에 대해서도 동일한 방식의 차이 변수(`_fix_mean_diff`)를 생성하여 시간대별 가격 격차를 수치화했습니다.

3.  **최대 월별 가격 차이 (Max Monthly Difference)**:
    - 매월 시간대별 가격 차이를 계산한 뒤, 고객별로 그 차이의 **최대값**을 추출했습니다.
    - 예: `off_peak_peak_var_max_monthly_diff`
    - 이는 연중 가장 큰 가격 격차가 발생했을 때의 충격(Price Shock)을 반영하기 위한 시도입니다.

### 3.2 계약 유지 기간 (Tenure)
- `date_end` (계약 종료일)와 `date_activ` (계약 시작일)의 차이를 계산하여 고객이 서비스를 이용한 기간인 `tenure` (연 단위)를 생성했습니다.
- `tenure`별 이탈률을 그룹화하여 분석함으로써, 장기 고객과 단기 고객 간의 이탈 성향 차이를 파악할 수 있는 기반을 마련했습니다.

### 3.3 날짜 데이터 변환 (Transforming Dates into Months)
- 모델이 날짜 형식을 직접 학습하기 어렵기 때문에, 기준 시점(`2016-01-01`)으로부터의 **개월 수(Months)**로 변환했습니다.
- `convert_months` 함수를 정의하여 다음 변수들을 생성:
    - `months_activ`: 계약 활성 기간 (월).
    - `months_to_end`: 계약 종료까지 남은 기간 (월).
    - `months_modif_prod`: 마지막 상품 변경 후 경과 기간 (월).
    - `months_renewal`: 마지막 갱신 후 경과 기간 (월).
- 변환 후 원본 날짜 컬럼은 제거하여 다중공선성 문제를 방지하고 데이터 차원을 축소했습니다.

### 3.4 불리언 데이터 변환 (Transforming Boolean Data)
- `has_gas` 컬럼의 값을 문자열('t', 'f')에서 수치형(1, 0)으로 변환했습니다.
- 이는 대부분의 머신러닝 알고리즘이 수치형 입력을 요구하기 때문입니다.

### 3.5 범주형 데이터 인코딩 (Categorical Encoding)
- `channel_sales` (판매 채널) 및 `origin_up` (가입 경로) 컬럼에 대해 **One-Hot Encoding**을 수행했습니다 (`pd.get_dummies` 사용).
- **차원 축소**: 생성된 더미 변수 중 빈도가 낮은 일부 채널이나 의미 없는 컬럼(예: `MISSING` 등)을 제거하여 모델의 복잡도를 낮추고 과적합(Overfitting)을 방지하려 했습니다.

### 3.6 수치형 데이터 변환 (Transforming Numerical Data)
- **왜도(Skewness) 처리**: 데이터 분포가 한쪽으로 치우친(Skewed) 수치형 변수들을 식별했습니다.
    - 대상 변수: `cons_12m`, `cons_gas_12m`, `cons_last_month`, `forecast_cons_12m`, `forecast_cons_year`, `forecast_meter_rent_12m`, `imp_cons`.
- **로그 변환 (Log Transformation)**: `np.log10(x + 1)`을 적용하여 데이터 분포를 정규분포에 가깝게 만들었습니다. `+1`을 더해주는 이유는 0값에 대한 로그 연산 오류(-inf)를 방지하기 위함입니다.
- 이를 통해 이상치(Outlier)의 영향을 줄이고 선형 모델 등의 성능 향상을 도모했습니다.

### 3.7 상관관계 분석 (Correlation Analysis)
- 전처리가 완료된 데이터프레임에 대해 상관계수 행렬(`df.corr()`)을 계산하고, 이를 Plotly Heatmap 등으로 시각화했습니다.
- 변수들 간의 다중공선성을 확인하고, 타겟 변수(`churn`)와 상관관계가 높은 주요 변수를 식별하는 단계입니다.

## 4. 결론 및 요약
이 노트북은 원시 데이터에서 모델 학습에 적합한 형태의 데이터셋을 구축하는 전 과정을 체계적으로 수행했습니다.
1.  **시계열 특성 반영**: 단순한 날짜 데이터를 '기간(Months)'과 '가격 변동(Difference)'이라는 의미 있는 수치형 변수로 변환했습니다.
2.  **데이터 정규화**: 왜도가 심한 소비 데이터에 로그 변환을 적용하여 분포를 개선했습니다.
3.  **범주형 변수 처리**: One-Hot Encoding을 통해 기계가 이해할 수 있는 형태로 변환했습니다.

이러한 전처리 과정은 후속 모델링 단계(예: LightGBM, Random Forest 등)에서 모델이 데이터의 패턴을 더 효과적으로 학습하고, 결과적으로 이탈 예측 정확도를 높이는 데 기여할 것으로 예상됩니다.
