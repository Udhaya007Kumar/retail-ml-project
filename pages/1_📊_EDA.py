# ============================================================
# PAGE: EDA — Exploratory Data Analysis  📊
# ============================================================
# 6 Plots:
#   1. Sales Distribution (Histogram)
#   2. Sales by Season (Box Plot)
#   3. Correlation Heatmap
#   4. Sales vs Customer Count (Scatter)
#   5. Region-wise Sales (Bar)
#   6. Daily Sales Heatmap (Calendar)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import sys
sys.path.insert(0, 'src')
from model_preparation import load_data, set_page_style

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
set_page_style()

df = load_data()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 📊 EDA Settings")
    selected_category = st.multiselect(
        "Filter by Category",
        options=df['Category'].dropna().unique().tolist(),
        default=df['Category'].dropna().unique().tolist()
    )
    selected_region = st.multiselect(
        "Filter by Region",
        options=df['Region'].dropna().unique().tolist(),
        default=df['Region'].dropna().unique().tolist()
    )

# Filter apply
df_filtered = df[
    df['Category'].isin(selected_category) &
    df['Region'].isin(selected_region)
]

# ── Header ──
st.markdown("""
<h1 style='color:#60a5fa'>📊 Exploratory Data Analysis</h1>
<p style='color:#888'>Visualize patterns in your retail sales data — 6 interactive plots</p>
<hr style='border-color:#333'>
""", unsafe_allow_html=True)

# Quick stats
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Records", f"{len(df_filtered):,}")
c2.metric("Avg Sales",     f"₹{df_filtered['Sales'].mean():,.0f}")
c3.metric("Max Sales",     f"₹{df_filtered['Sales'].max():,}")
c4.metric("Min Sales",     f"₹{df_filtered['Sales'].min():,}")

st.markdown("---")

# ============================================================
# PLOT 1 + 2
# ============================================================
col1, col2 = st.columns(2)

# ── Plot 1: Sales Distribution ──
with col1:
    st.markdown("#### 📉 Plot 1 — Sales Distribution")
    st.caption("Sales values எப்படி distribute ஆகுதுன்னு காட்டுகிறது")
    fig1 = px.histogram(
        df_filtered, x='Sales',
        nbins=50,
        color_discrete_sequence=['#a78bfa'],
        template='plotly_dark',
        labels={'Sales': 'Sales Amount (₹)', 'count': 'Frequency'}
    )
    fig1.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', bargap=0.05,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    fig1.add_vline(
        x=df_filtered['Sales'].mean(),
        line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Mean: ₹{df_filtered['Sales'].mean():,.0f}",
        annotation_font_color="#f59e0b"
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Plot 2: Sales by Season ──
with col2:
    st.markdown("#### 🌿 Plot 2 — Sales by Season (Box Plot)")
    st.caption("Season மாறும்போது sales எப்படி மாறுதுன்னு காட்டுகிறது")
    fig2 = px.box(
        df_filtered, x='Season', y='Sales',
        color='Season',
        color_discrete_sequence=['#34d399','#f59e0b','#60a5fa','#f87171'],
        template='plotly_dark',
        points="outliers"
    )
    fig2.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white', showlegend=False,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PLOT 3 + 4
# ============================================================
col3, col4 = st.columns(2)

# ── Plot 3: Correlation Heatmap ──
with col3:
    st.markdown("#### 🔥 Plot 3 — Correlation Heatmap")
    st.caption("எந்த features-க்கு Sales-ஓட strong relationship இருக்குன்னு காட்டுகிறது")

    num_cols = ['Sales', 'Quantity', 'Customer_Count', 'Month',
                'Is_Weekend', 'Is_Holiday', 'Has_Promotion',
                'Max_Discount', 'Total_Stock', 'Store_Age_Years']
    num_cols = [c for c in num_cols if c in df_filtered.columns]

    corr_matrix = df_filtered[num_cols].corr().round(2)

    fig3 = go.Figure(data=go.Heatmap(
        z     = corr_matrix.values,
        x     = corr_matrix.columns.tolist(),
        y     = corr_matrix.columns.tolist(),
        colorscale = 'RdBu',
        zmid  = 0,
        text  = corr_matrix.values.round(2),
        texttemplate = "%{text}",
        textfont={"size":10},
        hoverongaps = False
    ))
    fig3.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white',
        margin=dict(l=0,r=0,t=20,b=0),
        height=380
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Plot 4: Sales vs Customer Count ──
with col4:
    st.markdown("#### 👥 Plot 4 — Sales vs Customer Count")
    st.caption("Customer அதிகமா வந்தா Sales அதிகமா ஆகுதான்னு காட்டுகிறது")

    sample = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)
    fig4 = px.scatter(
        sample, x='Customer_Count', y='Sales',
        color='Category',
        size='Quantity',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template='plotly_dark',
        trendline='ols'
    )
    fig4.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white',
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# PLOT 5 + 6
# ============================================================
col5, col6 = st.columns(2)

# ── Plot 5: Region-wise Sales ──
with col5:
    st.markdown("#### 🗺️ Plot 5 — Region-wise Sales")
    st.caption("எந்த region-ல sales அதிகமா இருக்குன்னு காட்டுகிறது")

    region_data = df_filtered.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    fig5 = px.bar(
        region_data, x='Region', y='Sales',
        color='Category',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template='plotly_dark',
        text_auto='.2s'
    )
    fig5.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white',
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── Plot 6: Sales by Day of Week & Month (Heatmap) ──
with col6:
    st.markdown("#### 📅 Plot 6 — Day × Month Sales Heatmap")
    st.caption("எந்த day, எந்த month-ல sales அதிகமா இருக்குன்னு காட்டுகிறது")

    pivot = df_filtered.pivot_table(
        values='Sales', index='DayOfWeek', columns='Month', aggfunc='mean'
    ).fillna(0)

    day_labels   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

    pivot.index   = [day_labels[i] for i in pivot.index if i < 7]
    pivot.columns = [month_labels[i-1] for i in pivot.columns if 1 <= i <= 12]

    fig6 = px.imshow(
        pivot,
        color_continuous_scale='Viridis',
        template='plotly_dark',
        aspect='auto',
        labels=dict(color="Avg Sales")
    )
    fig6.update_layout(
        plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
        font_color='white',
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Key Insights ──
st.markdown("---")
st.markdown("### 💡 Key Insights from EDA")

i1, i2, i3 = st.columns(3)
with i1:
    best_season = df_filtered.groupby('Season')['Sales'].mean().idxmax()
    st.success(f"🌿 **Best Season:** {best_season} has highest average sales")
with i2:
    best_region = df_filtered.groupby('Region')['Sales'].sum().idxmax()
    st.info(f"🗺️ **Top Region:** {best_region} leads in total sales")
with i3:
    promo_avg   = df_filtered[df_filtered['Has_Promotion']==1]['Sales'].mean()
    no_promo    = df_filtered[df_filtered['Has_Promotion']==0]['Sales'].mean()
    diff        = ((promo_avg - no_promo) / no_promo * 100)
    st.warning(f"🎯 **Promotion Impact:** {diff:+.1f}% sales change with promotion")
