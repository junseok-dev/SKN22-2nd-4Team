# 데이터 컬럼 설명서 (Data Column Description)

본 문서는 `data/01_raw` 디렉토리에 위치한 원본 데이터 파일들의 컬럼에 대한 상세 설명을 담고 있습니다.

---

## 1. `client_data.csv` (고객 데이터)
고객의 계약 정보, 소비량, 예측값, 이탈 여부 등을 포함하는 주요 데이터셋입니다.

### 식별자 및 기본 정보
- **`id`**: 고객 고유 식별자 (Unique Identifier).
- **`channel_sales`**: 판매 채널 코드 (고객이 유입된 경로).
- **`origin_up`**: 고객이 처음 가입한 전기 캠페인 코드.
- **`nb_prod_act`**: 현재 활성화된 제품 및 서비스의 개수.
- **`has_gas`**: 가스 서비스 이용 여부 (`t`: True, `f`: False).
- **`num_years_antig`**: 고객의 서비스 이용 연수 (Antiquity).

### 날짜 관련 (Dates)
- **`date_activ`**: 계약 활성화 날짜 (가입일).
- **`date_end`**: 등록된 계약 종료 날짜.
- **`date_modif_prod`**: 마지막 제품/서비스 변경 날짜.
- **`date_renewal`**: 다음 계약 갱신 날짜.

### 소비량 데이터 (Consumption)
- **`cons_12m`**: 지난 12개월 동안의 전기 소비량.
- **`cons_gas_12m`**: 지난 12개월 동안의 가스 소비량.
- **`cons_last_month`**: 지난달 전기 소비량.
- **`imp_cons`**: 현재 납부된 소비량 (Current paid consumption).

### 예측 데이터 (Forecasts)
- **`forecast_cons_12m`**: 향후 12개월 예상 전기 소비량.
- **`forecast_cons_year`**: 다음 연도 예상 전기 소비량 (Calendar Year).
- **`forecast_discount_energy`**: 현재 적용된 에너지 할인 예측값.
- **`forecast_meter_rent_12m`**: 향후 12개월 계량기 렌탈 비용 예측값.
- **`forecast_price_energy_off_peak`**: 1구간(Off-peak) 에너지 가격 예측값.
- **`forecast_price_energy_peak`**: 2구간(Peak) 에너지 가격 예측값.
- **`forecast_price_pow_off_peak`**: 1구간(Off-peak) 전력(Power) 가격 예측값.

### 마진 및 전력량 (Margins & Power)
- **`margin_gross_pow_ele`**: 전력 구독에 대한 총 마진 (Gross Margin).
- **`margin_net_pow_ele`**: 전력 구독에 대한 순 마진 (Net Margin).
- **`net_margin`**: 총 순 마진 (Total Net Margin).
- **`pow_max`**: 계약된 최대 전력 용량 (Subscribed Power).

### 타겟 변수 (Target)
- **`churn`**: 이탈 여부 (1: 이탈, 0: 유지). 향후 3개월 내 이탈 여부를 나타냄.

---

## 2. `price_data.csv` (가격 데이터)
각 고객별, 시점별 에너지 및 전력 가격 변동 정보를 담고 있습니다.

### 기본 정보
- **`id`**: 고객 고유 식별자 (`client_data.csv`의 id와 매핑).
- **`price_date`**: 가격 데이터의 기준 날짜 (월별 데이터).

### 변동 요금 (Variable Prices - Energy)
에너지 사용량에 따라 부과되는 요금입니다.
- **`price_off_peak_var`**: 1구간(Off-peak) 에너지 가격.
- **`price_peak_var`**: 2구간(Peak) 에너지 가격.
- **`price_mid_peak_var`**: 3구간(Mid-peak) 에너지 가격.

### 고정 요금 (Fixed Prices - Power)
전력 용량 등에 따라 고정적으로 부과되는 요금입니다.
- **`price_off_peak_fix`**: 1구간(Off-peak) 전력 가격.
- **`price_peak_fix`**: 2구간(Peak) 전력 가격.
- **`price_mid_peak_fix`**: 3구간(Mid-peak) 전력 가격.

---
**참고**: 'Off-peak', 'Peak', 'Mid-peak'는 전력 사용 시간대에 따른 요금 구간을 의미합니다. 통상적으로 Off-peak가 가장 저렴하고 Peak가 가장 비싼 구간입니다.
