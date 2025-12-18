import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix


# CatBoost는 설치 여부에 따라 조건부 임포트 (파일이 없거나 라이브러리 부재 시 에러 방지)
try:
    from catboost import CatBoostClassifier
    model_available = True
except ImportError:
    model_available = False
# -----------------------------------------------------------------------------
# 0. 한글 폰트 및 시각화 설정 (Matplotlib)
# -----------------------------------------------------------------------------
def set_korean_font():
    # 운영체제별 폰트 자동 설정
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    else:
        # 리눅스 등 기타 환경에서는 기본 폰트 사용 (한글이 깨질 경우 영문으로 표기 권장)
        plt.rc('font', family='sans-serif')
    
    plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지

set_korean_font()

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 스타일링
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Diagnosis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 기업용 커스텀 CSS 적용
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .big-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .loss-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e74c3c;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 
# 
# ----------------------------------------------
# 2. 데이터 로드 및 전처리 함수
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # 데이터 경로 설정 (업로드된 파일 기준)
    DATA_PATH = "data/01_raw/train.csv"
    if not os.path.exists(DATA_PATH):
        st.error(f"데이터 파일({DATA_PATH})을 찾을 수 없습니다.")
        return None

    df = pd.read_csv(DATA_PATH)

    # 전처리 (generate_plots.py 로직 유지)
    if 'international_plan' in df.columns:
        df['international_plan'] = (df['international_plan'] == 'yes').astype(int)
    if 'voice_mail_plan' in df.columns:
        df['voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int)
    
    # Target 변환
    if 'churn' in df.columns and df['churn'].dtype == object:
        df['churn'] = df['churn'].apply(lambda x: 1 if x == 'yes' else 0)

    # 파생 변수: 총 매출 (Revenue) 추정
    # Total Charge 컬럼들의 합을 월 매출로 가정
    charge_cols = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge']
    df['total_revenue'] = df[charge_cols].sum(axis=1)
    
    return df

@st.cache_resource
def load_model():
    model_path = "churn_model.cbm"
    if model_available and os.path.exists(model_path):
        try:
            model = CatBoostClassifier()
            model.load_model(model_path)
            return model
        except:
            return None
    return None

df = load_data()
model = load_model()

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바 네비게이션
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️Churn Diagnosis")
st.sidebar.info("고객이탈 진단 방지")
page = st.sidebar.radio("section", ["1. 현황 진단", "2. 솔루션 & 시뮬레이션", "3. 기대 효과"])

st.sidebar.markdown("---")
currency_symbol = st.sidebar.text_input("화폐 단위", value="$")
st.sidebar.markdown("📝 프로젝트 목표")
st.sidebar.markdown("현재 회사에")
st.sidebar.markdown("1.고객이탈에 진단을 통해")
st.sidebar.markdown("2.최적에 솔루션을 제공하고")
st.sidebar.markdown("3.기업에 매출 증대")


# -----------------------------------------------------------------------------
# 4. 페이지별 로직
# -----------------------------------------------------------------------------

# === Page 1: 현황 진단 ===
if page == "1. 현황 진단":
    st.title("🩺 고객 이탈 현황 및 재무적 손실 진단")
    st.markdown("현재 회사의 고객 이탈 현황과 그로 인한 **직접적인 재무 손실**을 시각화합니다.")

    # KPI Calculation
    total_customers = len(df)
    churn_count = df['churn'].sum()
    churn_rate = churn_count / total_customers * 100
    total_revenue = df['total_revenue'].sum()
    lost_revenue = df[df['churn'] == 1]['total_revenue'].sum()

    # Top KPI Display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>총 고객 수</h3><div class='big-number'>{total_customers:,.0f}명</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>이탈률 (Churn Rate)</h3><div class='loss-number'>{churn_rate:.1f}%</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>총 매출 (Monthly)</h3><div class='big-number'>{currency_symbol}{total_revenue:,.0f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>이탈로 인한 손실</h3><div class='loss-number'>{currency_symbol}{lost_revenue:,.0f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚨 핵심 이탈 원인 및 손실 분석")
    st.markdown("회사의 수익을 갉아먹는 **주요 이탈 원인**과 각 원인별 **구체적인 손실액**입니다.")

    # Header Row
    st.markdown("""
    <div style="display: flex; justify-content: space-between; padding: 10px; border-bottom: 2px solid #ddd; font-weight: bold; color: #555;">
        <div style="width: 40%;">📌 이탈 원인 (Risk Factor)</div>
        <div style="width: 30%; text-align: center;">📉 이탈률 (Churn Rate)</div>
        <div style="width: 30%; text-align: right;">💸 손실액 (Revenue Loss)</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. 고객센터 전화 연결 (CS Calls >= 4)
    cs_risk_group = df[df['number_customer_service_calls'] >= 4]
    cs_churn_rate = cs_risk_group['churn'].mean() * 100 if len(cs_risk_group) > 0 else 0
    cs_loss = cs_risk_group[cs_risk_group['churn'] == 1]['total_revenue'].sum()

    st.markdown(f"""
    <div class='risk-row' style="display: flex; align-items: center; justify-content: space-between;">
        <div style="width: 40%;">
            <div class='risk-title'>① 고객센터 전화 연결</div>
            <div style="font-size: 0.9em; color: gray;">(고객센터 통화 4회 이상 악성 불만 고객)</div>
        </div>
        <div style="width: 30%; text-align: center;">
            <div class='risk-stat' style="color: #e74c3c;">{cs_churn_rate:.1f}%</div>
            <div style="font-size: 0.8em; color: gray;">Avg: {churn_rate:.1f}%</div>
        </div>
        <div style="width: 30%; text-align: right;">
            <div class='risk-stat'>{currency_symbol}{cs_loss:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. 국제전화 통화량 (International Plan/Usage)
    intl_risk_group = df[df['international_plan'] == 1]
    intl_churn_rate = intl_risk_group['churn'].mean() * 100 if len(intl_risk_group) > 0 else 0
    intl_loss = intl_risk_group[intl_risk_group['churn'] == 1]['total_revenue'].sum()

    st.markdown(f"""
    <div class='risk-row' style="display: flex; align-items: center; justify-content: space-between;">
        <div style="width: 40%;">
            <div class='risk-title'>② 국제전화 통화량</div>
            <div style="font-size: 0.9em; color: gray;">(국제전화 플랜 가입 및 고사용자군)</div>
        </div>
        <div style="width: 30%; text-align: center;">
            <div class='risk-stat' style="color: #e74c3c;">{intl_churn_rate:.1f}%</div>
            <div style="font-size: 0.8em; color: gray;">Avg: {churn_rate:.1f}%</div>
        </div>
        <div style="width: 30%; text-align: right;">
            <div class='risk-stat'>{currency_symbol}{intl_loss:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# === Page 2: 솔루션 & 시뮬레이션 ===
elif page == "2. 솔루션 & 시뮬레이션":
    st.title("💊 이탈 방지 솔루션 & 시뮬레이터")
    st.markdown("데이터 기반의 대안을 적용했을 때 예상되는 효과를 시뮬레이션합니다.")

    # 레이아웃: 왼쪽(시뮬레이터) / 오른쪽(결과)
    col_simulator, col_results = st.columns([1, 1])

    # --- 1. 왼쪽: 시뮬레이터 조작 ---
    with col_simulator:
        st.markdown("### 🎛️ 전략 시뮬레이터")
        st.info("아래 슬라이더를 조절하여 예상 방어율을 설정하세요.")
        
        st.markdown("**1️⃣ 케어 프로그램 (CS 집중 관리)**")
        st.caption("대상: 고객센터 전화 **3회 이상** 시도한 잠재 불만 고객")
        improvement_cs = st.slider("케어 성공률 (예상 방어율 %)", 0, 100, 30, step=5)
        
        st.write("") 
        
        st.markdown("**2️⃣ 국제전화 요금제 개편**")
        st.caption("대상: 국제전화 플랜 가입자 및 고사용자")
        improvement_intl = st.slider("요금제 개편 성공률 (예상 방어율 %)", 0, 100, 15, step=5)

    # --- 시뮬레이션 로직 ---
    df_sim = df.copy()
    
    # Logic 1: CS Calls >= 3
    high_risk_cs_indices = df_sim[(df_sim['number_customer_service_calls'] >= 3) & (df_sim['churn'] == 1)].index
    saved_cs_count = int(len(high_risk_cs_indices) * (improvement_cs / 100))
    if saved_cs_count > 0:
        saved_indices = np.random.choice(high_risk_cs_indices, saved_cs_count, replace=False)
        df_sim.loc[saved_indices, 'churn'] = 0
        
    # Logic 2: International Plan
    high_risk_intl_indices = df_sim[(df_sim['international_plan'] == 1) & (df_sim['churn'] == 1)].index
    saved_intl_count = int(len(high_risk_intl_indices) * (improvement_intl / 100))
    if saved_intl_count > 0:
        saved_indices_intl = np.random.choice(high_risk_intl_indices, saved_intl_count, replace=False)
        df_sim.loc[saved_indices_intl, 'churn'] = 0

    # 결과 계산
    new_lost_revenue = df_sim[df_sim['churn'] == 1]['total_revenue'].sum()
    original_lost_revenue = df[df['churn'] == 1]['total_revenue'].sum()
    recovered_revenue = original_lost_revenue - new_lost_revenue
    
    new_churn_rate = df_sim['churn'].mean() * 100
    original_churn_rate = df['churn'].mean() * 100

    # --- 2. 오른쪽: 시뮬레이션 결과 (Matplotlib로 대체하여 에러 방지) ---
    with col_results:
        st.markdown("### 🚀 시뮬레이션 결과")
        st.markdown("전략 적용 시 예상되는 **수치적 변화**입니다.")
        
        # 메트릭
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="📉 예상 이탈률", 
                      value=f"{new_churn_rate:.2f}%", 
                      delta=f"{new_churn_rate - original_churn_rate:.2f}%p",
                      delta_color="inverse")
        with m_col2:
            st.metric(label="💰 월 매출 회복액", 
                      value=f"{currency_symbol}{recovered_revenue:,.0f}", 
                      delta=f"+ {(recovered_revenue/original_lost_revenue)*100:.1f}% 회복",
                      delta_color="normal")
        
        st.write("")
        st.markdown("**📊 이탈률 변화 (Before vs After)**")
        
        # [수정됨] Matplotlib 그래프 (Altair 에러 해결)
        fig, ax = plt.subplots(figsize=(6, 4))
        x_labels = ['Before (현재)', 'After (개선후)']
        y_values = [original_churn_rate, new_churn_rate]
        colors = ['#95a5a6', '#e74c3c'] # 회색 -> 빨강
        
        bars = ax.bar(x_labels, y_values, color=colors, width=0.5)
        
        # 디자인
        ax.set_ylabel('이탈률 (%)')
        ax.set_ylim(0, max(y_values)*1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 바 위에 숫자 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        st.pyplot(fig)

    # --- 3. 하단: 구체적 솔루션 (안전한 박스 디자인 사용) ---
    st.markdown("---")
    st.subheader("💡 구체적인 실행 솔루션 (Action Plan)")
    
    col_sol1, col_sol2 = st.columns(2)
    
    with col_sol1:
        st.info("📞 1. 케어 프로그램 (Care Program)")
        st.markdown("""
        **🎯 대 상:** 최근 고객센터 연결 **3회 이상** 시도한 고위험 고객
        
        **📋 내 용:**
        - 상담 연결 전 **"우수 상담사 우선 배정"**
        - 통화 종료 후 **"불편 해결 확인 해피콜"** 자동 예약
        - 다음 달 기본료 **10% 할인 쿠폰** 즉시 발송
        """)
        
    with col_sol2:
        st.success("🌐 2. 글로벌 커넥트 요금제 (Retention)")
        st.markdown("""
        **🎯 대 상:** 국제전화 사용량이 많거나 전용 플랜 가입자
        
        **📋 내 용:**
        - 사용량 구간별 **자동 할인율(Tiered Discount)** 적용
        - 해외 체류 가족 등록 시 **"패밀리 무료 통화(30분)"** 제공
        - 경쟁사 대비 5% 저렴한 **장기 약정(1년)** 제안
        """)
# === Page 3: 기대 효과 ===
elif page == "3. 기대 효과":
    st.title("📈 변화된 회사의 모습")
    st.markdown("제안된 전략을 모두 실행했을 때, 회사가 얻게 될 **최종적인 모습(To-Be)**입니다.")

    # 저장된 시뮬레이션 값 (없으면 기본값 가정)
    # 여기서는 예시를 위해 고정된 '성공 시나리오'를 보여줍니다.
    
    # 가정: 종합적인 전략 실행으로 이탈률 14% -> 8% 감소 가정
    current_churn = 14.1
    target_churn = 8.5
    
    current_loss = 39188  # 계산된 값
    projected_loss = current_loss * (target_churn / current_churn)
    annual_recovery = (current_loss - projected_loss) * 12 # 연간 환산

    st.markdown("### 🏆 Executive Summary")
    
    col_final1, col_final2, col_final3 = st.columns(3)
    
    with col_final1:
        st.warning("Current (AS-IS)")
        st.markdown(f"**이탈률:** {current_churn}%")
        st.markdown(f"**월 손실:** {currency_symbol}{current_loss:,.0f}")
        
    with col_final2:
        st.success("Projected (TO-BE)")
        st.markdown(f"**이탈률:** {target_churn}%")
        st.markdown(f"**월 손실:** {currency_symbol}{projected_loss:,.0f}")
        
    with col_final3:
        st.info("Net Impact (Yearly)")
        st.markdown(f"**이탈률 개선:** -{(current_churn - target_churn):.1f}%p")
        st.markdown(f"**연간 매출 증대:** {currency_symbol}{annual_recovery:,.0f}")

    st.markdown("---")
    st.subheader("📊 연간 매출 회복 시각화")
    
    # Waterfall chart data structure
    impact_data = pd.DataFrame({
        'Category': ['현재 연간 손실', 'CS 개선 효과', '요금제 개편 효과', '기타 마케팅 효과', '최종 잔존 손실'],
        'Amount': [-current_loss*12, 
                   (current_loss*12)*0.15, 
                   (current_loss*12)*0.10, 
                   (current_loss*12)*0.05, 
                   0] # 마지막은 계산
    })
    impact_data.iloc[4, 1] = impact_data['Amount'].sum() # 잔존 손실은 음수로 표현되어야 하므로 조정 필요하지만, 시각적 표현을 위해 단순화
    
    # 간단한 Bar chart로 표현 (Waterfall 대신 이해하기 쉽게)
    comparison_df = pd.DataFrame({
        'Status': ['현재 (AS-IS)', '전략 적용 후 (TO-BE)'],
        'Annual Loss': [current_loss*12, projected_loss*12]
    })
    
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    sns.barplot(y='Status', x='Annual Loss', data=comparison_df, palette=['#e74c3c', '#2ecc71'], ax=ax3)
    ax3.set_xlabel("연간 이탈 손실액 ($)")
    
    # 텍스트 주석 추가
    for i, v in enumerate(comparison_df['Annual Loss']):
        ax3.text(v + 1000, i, f"${v:,.0f}", va='center', fontweight='bold')
        
    st.pyplot(fig3)
    
    st.markdown("""
    ### 📝 최종 제언
    > "고객 이탈은 막을 수 없는 자연재해가 아닙니다. 데이터 기반의 **정밀한 타겟팅(Targeting)**과 **적절한 오퍼(Offer)**가 있다면,
    > 연간 **${:,.0f}** 규모의 매출을 추가로 확보할 수 있습니다. 지금 바로 솔루션을 도입하십시오."
    """.format(annual_recovery))