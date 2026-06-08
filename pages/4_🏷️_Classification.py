# ============================================================
# PAGE: CLASSIFICATION  🏷️
# ============================================================
# Sales High / Medium / Low-ஆ இருக்கான்னு predict பண்றோம்
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, 'src')
from model_preparation import load_data, load_all_models, set_page_style, FEATURE_COLS

st.set_page_config(page_title="Classification", page_icon="🏷️", layout="wide")
set_page_style()

df     = load_data()
models = load_all_models()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🏷️ Classification")
    st.markdown("High / Medium / Low sales predict")
    st.markdown("---")
    st.markdown("**Model Accuracy:**")
    st.markdown("- Logistic Reg: 90.8%")
    st.markdown("- Random Forest: 88.0%")
    st.markdown("- **XGBoost: 99.4% ⭐**")

# ── Header ──
st.markdown("""
<h1 style='color:#34d399'>🏷️ Sales Classification</h1>
<p style='color:#888'>Sales High / Medium / Low category predict பண்றோம்</p>
<hr style='border-color:#333'>
""", unsafe_allow_html=True)

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("### 🎛️ Enter Details")

c1, c2, c3, c4 = st.columns(4)
with c1:
    month        = st.selectbox("📅 Month", list(range(1,13)),
                                format_func=lambda x: ['Jan','Feb','Mar','Apr','May',
                                'Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    day_of_week  = st.selectbox("📆 Day", [0,1,2,3,4,5,6],
                                format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
with c2:
    customer_count = st.slider("👥 Customer Count", 1, 100, 30)
    has_promotion  = st.selectbox("🎯 Promotion?", [0,1], format_func=lambda x: "Yes" if x else "No")
with c3:
    max_discount   = st.slider("💸 Discount %", 0, 50, 10)
    store_age      = st.slider("🏪 Store Age", 1, 20, 5)
with c4:
    total_stock    = st.slider("📦 Stock", 0, 1000, 200)
    profit_margin  = st.slider("💹 Profit Margin %", 0.0, 100.0, 45.0)

c5, c6, c7, c8 = st.columns(4)
with c5:
    is_weekend   = st.selectbox("🏖️ Weekend?", [0,1], format_func=lambda x: "Yes" if x else "No")
    is_holiday   = st.selectbox("🎉 Holiday?", [0,1], format_func=lambda x: "Yes" if x else "No")
with c6:
    season_enc   = st.selectbox("🌿 Season", [0,1,2,3],
                                format_func=lambda x: ['Autumn','Monsoon','Summer','Winter'][x])
    category_enc = st.selectbox("🏷️ Category", [0,1,2,3,4],
                                format_func=lambda x: ['Electronics','Food','Sports','Clothing','Home'][x])
with c7:
    store_type_enc = st.selectbox("🏬 Store Type", [0,1,2,3],
                                  format_func=lambda x: ['Mall','Supermarket','Outlet','Mini'][x])
    region_enc   = st.selectbox("🗺️ Region", [0,1],
                                format_func=lambda x: ['North','South'][x])
with c8:
    brand_enc      = st.selectbox("🔖 Brand", list(range(7)),
                                  format_func=lambda x: f"Brand {x+1}")
    quarter        = st.selectbox("🗓️ Quarter", [1,2,3,4])

# ============================================================
# PREDICT
# ============================================================
st.markdown("---")
predict_btn = st.button("🔮 Predict Sales Category", use_container_width=True)

LABEL_MAP  = {0: "🔴 Low", 1: "🟡 Medium", 2: "🟢 High"}
COLOR_MAP  = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}
ADVICE_MAP = {
    0: "Sales குறைவா இருக்கு — Promotion try பண்ணுங்க!",
    1: "Sales average-ஆ இருக்கு — கொஞ்சம் improve பண்ணலாம்.",
    2: "Excellent! Sales மிகவும் நல்லா இருக்கு! 🎉"
}

if predict_btn:
    input_data = pd.DataFrame([{
        'Month': month, 'DayOfWeek': day_of_week, 'Quarter': quarter,
        'Is_Weekend': is_weekend, 'Is_Holiday': is_holiday,
        'Season_Encoded': season_enc, 'Category_Encoded': category_enc,
        'Brand_Encoded': brand_enc, 'Store_Type_Encoded': store_type_enc,
        'Region_Encoded': region_enc, 'Price_Range_Encoded': 2,
        'Payment_Method_Encoded': 1, 'Has_Promotion': has_promotion,
        'Max_Discount': max_discount, 'Store_Age_Years': store_age,
        'Total_Stock': total_stock, 'Customer_Count': customer_count,
        'Profit_Margin': profit_margin, 'Revenue_Per_Customer': 3500.0,
        'Avg_Discount': max_discount * 0.8,
        'Stock_Available': 1 if total_stock > 0 else 0,
        'Discount_Sales_Ratio': max_discount / 5000.0,
    }])

    scaler   = models['scaler']
    input_sc = scaler.transform(input_data)

    # Predictions from 3 models
    lr_pred  = int(models['logistic_clf'].predict(input_sc)[0])
    rf_pred  = int(models['rf_clf'].predict(input_data)[0])
    xgb_pred = int(models['xgb_clf'].predict(input_data)[0])

    # Probabilities
    lr_prob  = models['logistic_clf'].predict_proba(input_sc)[0]
    rf_prob  = models['rf_clf'].predict_proba(input_data)[0]
    xgb_prob = models['xgb_clf'].predict_proba(input_data)[0]

    st.markdown("### 🎯 Prediction Results")
    r1, r2, r3 = st.columns(3)

    for col, name, pred, prob, acc in [
        (r1, "Logistic Regression", lr_pred,  lr_prob,  "90.8%"),
        (r2, "Random Forest",       rf_pred,  rf_prob,  "88.0%"),
        (r3, "XGBoost ⭐",          xgb_pred, xgb_prob, "99.4%"),
    ]:
        color = COLOR_MAP[pred]
        with col:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{color}22,{color}11);
            border-left:4px solid {color};border-radius:12px;
            padding:20px;text-align:center;margin-bottom:12px'>
            <p style='color:#888;margin:0;font-size:13px'>{name}</p>
            <p style='color:{color};font-size:2rem;font-weight:700;margin:8px 0'>
                {LABEL_MAP[pred]}
            </p>
            <p style='color:#555;font-size:12px'>Accuracy: {acc}</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            prob_df = pd.DataFrame({
                'Category': ['Low','Medium','High'],
                'Probability': prob * 100
            })
            fig = px.bar(
                prob_df, x='Category', y='Probability',
                color='Category',
                color_discrete_sequence=['#ef4444','#f59e0b','#22c55e'],
                template='plotly_dark', text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                font_color='white', showlegend=False, height=250,
                yaxis_range=[0,110], margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Advice
    st.markdown("---")
    st.markdown("### 💡 Business Advice")
    advice_col = COLOR_MAP[xgb_pred]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{advice_col}22,{advice_col}11);
    border:1px solid {advice_col}55;border-radius:12px;padding:20px'>
    <h4 style='color:{advice_col};margin:0'>XGBoost Recommendation</h4>
    <p style='color:white;margin:8px 0 0;font-size:1.1rem'>
        {ADVICE_MAP[xgb_pred]}
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Historical distribution
    st.markdown("---")
    st.markdown("#### 📊 Sales Category Distribution in Dataset")
    cat_counts = df['Sales_Category_Encoded'].value_counts().reset_index()
    cat_counts.columns = ['Category_Code', 'Count']
    cat_counts['Category'] = cat_counts['Category_Code'].map({0:'Low',1:'Medium',2:'High'})
    fig2 = px.pie(
        cat_counts, names='Category', values='Count',
        color='Category',
        color_discrete_map={'Low':'#ef4444','Medium':'#f59e0b','High':'#22c55e'},
        hole=0.4, template='plotly_dark'
    )
    fig2.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👆 Values enter பண்ணி **Predict Sales Category** button click பண்ணுங்க!")

    # Show class balance
    st.markdown("#### 📊 Dataset Class Balance")
    cat_map = {0:'Low', 1:'Medium', 2:'High'}
    df_cat = df['Sales_Category_Encoded'].map(cat_map).value_counts().reset_index()
    df_cat.columns = ['Category','Count']
    fig = px.bar(
        df_cat, x='Category', y='Count',
        color='Category',
        color_discrete_map={'Low':'#ef4444','Medium':'#f59e0b','High':'#22c55e'},
        template='plotly_dark', text='Count'
    )
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', showlegend=False,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
