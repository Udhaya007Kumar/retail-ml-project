# ============================================================
# MODEL PREPARATION  🔧
# ============================================================
# என்ன பண்றோம்:
#   - Data load பண்றோம்
#   - Models load பண்றோம்
#   - எல்லா pages-க்கும் common helper functions
#   - இந்த file import பண்ணா எல்லாமே ready ஆகும்
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st

# ============================================================
# PATHS
# ============================================================

DATA_PATH    = "data/processed/final_dataset.csv"
CLUSTER_PATH = "data/processed/store_clusters.csv"
MODELS_PATH  = "models/"

# ============================================================
# MODEL LOAD FUNCTION
# ============================================================

def load_model(filename):
    """
    pickle file படிச்சு model return பண்றது.
    @st.cache_resource → ஒரு முறை load ஆனா browser-ல cache ஆகும்
    மீண்டும் reload ஆகாது → app fast-ஆ run ஆகும்
    """
    path = os.path.join(MODELS_PATH, filename)
    with open(path, 'rb') as f:   # 'rb' = read binary
        return pickle.load(f)


@st.cache_resource
def load_all_models():
    """
    எல்லா trained models-ஐயும் ஒரே dict-ல return பண்றது.
    """
    models = {
        # Regression models
        'linear_reg'   : load_model("linear_regression.pkl"),
        'rf_reg'       : load_model("rf_regressor.pkl"),
        'xgb_reg'      : load_model("xgb_regressor.pkl"),

        # Classification models
        'logistic_clf' : load_model("logistic_regression.pkl"),
        'rf_clf'       : load_model("rf_classifier.pkl"),
        'xgb_clf'      : load_model("xgb_classifier.pkl"),

        # Clustering
        'kmeans'       : load_model("kmeans_clustering.pkl"),

        # Scaler
        'scaler'       : load_model("scaler.pkl"),
    }
    return models


# ============================================================
# DATA LOAD FUNCTION
# ============================================================

@st.cache_data
def load_data():
    """
    Final dataset படிக்கிறோம்.
    @st.cache_data → data ஒரு முறை load ஆனா cache ஆகும்
    """
    df = pd.read_csv(DATA_PATH)

    # Date column fix
    df['Date'] = pd.to_datetime(df['Date'])

    return df


@st.cache_data
def load_cluster_data():
    """Store cluster data படிக்கிறோம்."""
    return pd.read_csv(CLUSTER_PATH)


# ============================================================
# FEATURE COLUMNS
# ============================================================
# Train பண்ணும்போது use பண்ணிய exact same features
# இதை மாத்தா predict work ஆகாது!

FEATURE_COLS = [
    'Month', 'DayOfWeek', 'Quarter', 'Is_Weekend', 'Is_Holiday',
    'Season_Encoded', 'Category_Encoded', 'Brand_Encoded',
    'Store_Type_Encoded', 'Region_Encoded', 'Price_Range_Encoded',
    'Payment_Method_Encoded', 'Has_Promotion', 'Max_Discount',
    'Store_Age_Years', 'Total_Stock', 'Customer_Count',
    'Profit_Margin', 'Revenue_Per_Customer', 'Avg_Discount',
    'Stock_Available', 'Discount_Sales_Ratio'
]


# ============================================================
# METRIC CARD HELPER
# ============================================================

def metric_card(title, value, delta=None, color="#7F77DD"):
    """
    Colorful metric card உருவாக்கும் helper function.
    Streamlit-ல st.metric() இருக்கு, ஆனா இது more colorful!
    """
    delta_html = ""
    if delta:
        arrow = "▲" if float(str(delta).replace('%','').replace('+','')) > 0 else "▼"
        delta_color = "#22c55e" if "▲" in arrow else "#ef4444"
        delta_html = f'<p style="color:{delta_color};font-size:14px;margin:0">{arrow} {delta}</p>'

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
    ">
        <p style="color:#888;font-size:13px;margin:0;font-weight:500">{title}</p>
        <p style="font-size:28px;font-weight:700;margin:4px 0;color:{color}">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE CONFIG HELPER
# ============================================================

def set_page_style():
    """
    எல்லா pages-க்கும் common CSS styles.
    """
    st.markdown("""
    <style>
        /* Main background */
        .main { background-color: #0f0f1a; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }

        /* Headers */
        h1 { color: #a78bfa !important; }
        h2 { color: #60a5fa !important; }
        h3 { color: #34d399 !important; }

        /* Metric boxes */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #a78bfa !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #7c3aed, #2563eb);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 24px;
            font-weight: 600;
        }

        /* Selectbox */
        .stSelectbox { color: white; }

        /* Tab styling */
        .stTabs [data-baseweb="tab"] {
            color: #a78bfa;
            font-weight: 600;
        }

        /* Divider */
        hr { border-color: #333; }
    </style>
    """, unsafe_allow_html=True)
