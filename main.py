# ============================================================
# MAIN.PY  🚀
# ============================================================
# என்ன பண்றோம்:
#   இந்த ஒரே file run பண்ணா எல்லாமே நடக்கும்!
#   Step 1 → Step 2 → Step 3 → Results save
#
# Run: python main.py
# ============================================================

import os
import sys
import time

# src/ folder-ஐ Python path-ல add பண்றோம்
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster         import KMeans

from xgboost import XGBRegressor, XGBClassifier

from model_saver import (
    save_regression_results,
    save_classification_results,
    save_feature_importance,
    generate_comparison_report,
    save_model_with_meta
)

# ============================================================
# PATHS
# ============================================================

DATA_PATH   = "data/processed/final_dataset.csv"
MODELS_PATH = "models/"
RESULTS_PATH = "models/results/"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

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
# BANNER PRINT
# ============================================================

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║         🛒 RETAIL SALES ML PROJECT — MAIN RUNNER        ║
║                                                          ║
║   Step 1: Data Cleaning    (auto-skipped if done)        ║
║   Step 2: Feature Eng.     (auto-skipped if done)        ║
║   Step 3: Model Training   + Results Save                ║
║   Step 4: Launch Streamlit Dashboard                     ║
╚══════════════════════════════════════════════════════════╝
""")


# ============================================================
# STEP CHECK — Already done-ஆ?
# ============================================================

def check_steps_done():
    """
    Step 1, 2 already run ஆச்சான்னு check பண்றோம்.
    இல்லன்னா run பண்றோம்.
    """
    print("🔍 Checking previous steps...")

    # Step 1 check
    if not os.path.exists("data/processed/sales_cleaned.csv"):
        print("  ⚠️  Step 1 not done. Running data_cleaning.py...")
        os.system("python src/data_cleaning.py")
    else:
        print("  ✅ Step 1 already done (data_cleaning)")

    # Step 2 check
    if not os.path.exists("data/processed/final_dataset.csv"):
        print("  ⚠️  Step 2 not done. Running feature_engineering.py...")
        os.system("python src/feature_engineering.py")
    else:
        print("  ✅ Step 2 already done (feature_engineering)")


# ============================================================
# DATA LOAD & SPLIT
# ============================================================

def load_and_split():
    print("\n📂 Loading final dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Features present-ஆ check
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"   Features available: {len(available)}")

    X       = df[available].fillna(0)
    y_reg   = df['Sales'].fillna(df['Sales'].median())
    y_class = df['Sales_Category_Encoded'].fillna(1)

    # Split
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Scale
    scaler        = StandardScaler()
    X_train_sc    = scaler.fit_transform(X_train)
    X_test_sc     = scaler.transform(X_test)

    # Save scaler
    with open(MODELS_PATH + "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    return (X_train, X_test, X_train_sc, X_test_sc,
            y_reg_train, y_reg_test,
            y_clf_train, y_clf_test,
            available, df, scaler)


# ============================================================
# TRAIN ALL REGRESSION MODELS
# ============================================================

def train_regression(X_train, X_test, X_train_sc, X_test_sc,
                     y_train, y_test, feature_cols, scaler):
    print("\n" + "═" * 60)
    print("📈 TRAINING REGRESSION MODELS")
    print("═" * 60)

    reg_metrics = {}

    # ── Linear Regression ──
    print("\n  1️⃣  Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    m, _ = save_regression_results(lr, "linear_regression", X_test, y_test, scaler)
    save_model_with_meta(lr, "linear_regression", "regression", m)
    reg_metrics['Linear Regression'] = m

    # ── Random Forest ──
    print("\n  2️⃣  Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    m, _ = save_regression_results(rf, "rf_regressor", X_test, y_test)
    save_model_with_meta(rf, "rf_regressor", "regression", m)
    save_feature_importance(rf, feature_cols, "rf_regressor")
    reg_metrics['Random Forest'] = m

    # ── XGBoost ──
    print("\n  3️⃣  XGBoost Regressor...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1,
                       max_depth=6, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    m, _ = save_regression_results(xgb, "xgb_regressor", X_test, y_test)
    save_model_with_meta(xgb, "xgb_regressor", "regression", m)
    save_feature_importance(xgb, feature_cols, "xgb_regressor")
    reg_metrics['XGBoost'] = m

    best = max(reg_metrics, key=lambda x: reg_metrics[x]['R2'])
    print(f"\n  🏆 Best Regression: {best} (R²={reg_metrics[best]['R2']:.4f})")

    return reg_metrics


# ============================================================
# TRAIN ALL CLASSIFICATION MODELS
# ============================================================

def train_classification(X_train, X_test, X_train_sc, X_test_sc,
                          y_train, y_test, feature_cols, scaler):
    print("\n" + "═" * 60)
    print("🏷️  TRAINING CLASSIFICATION MODELS")
    print("═" * 60)

    clf_metrics = {}

    # ── Logistic Regression ──
    print("\n  1️⃣  Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    m, _ = save_classification_results(lr, "logistic_regression", X_test, y_test, scaler)
    save_model_with_meta(lr, "logistic_regression", "classification", m)
    clf_metrics['Logistic Regression'] = m

    # ── Random Forest ──
    print("\n  2️⃣  Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    m, _ = save_classification_results(rf, "rf_classifier", X_test, y_test)
    save_model_with_meta(rf, "rf_classifier", "classification", m)
    save_feature_importance(rf, feature_cols, "rf_classifier")
    clf_metrics['Random Forest'] = m

    # ── XGBoost ──
    print("\n  3️⃣  XGBoost Classifier...")
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                         max_depth=6, random_state=42, verbosity=0,
                         eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    m, _ = save_classification_results(xgb, "xgb_classifier", X_test, y_test)
    save_model_with_meta(xgb, "xgb_classifier", "classification", m)
    save_feature_importance(xgb, feature_cols, "xgb_classifier")
    clf_metrics['XGBoost'] = m

    best = max(clf_metrics, key=lambda x: clf_metrics[x]['Accuracy'])
    print(f"\n  🏆 Best Classification: {best} ({clf_metrics[best]['Accuracy']*100:.1f}%)")

    return clf_metrics


# ============================================================
# TRAIN CLUSTERING
# ============================================================

def train_clustering(df):
    print("\n" + "═" * 60)
    print("🔵 TRAINING CLUSTERING MODEL")
    print("═" * 60)

    from sklearn.cluster  import KMeans
    from sklearn.metrics  import silhouette_score

    store_agg = df.groupby('Store_ID').agg(
        Total_Sales    = ('Sales', 'sum'),
        Avg_Sales      = ('Sales', 'mean'),
        Total_Customers= ('Customer_Count', 'sum'),
        Avg_Quantity   = ('Quantity', 'mean'),
        Transactions   = ('Sale_ID', 'count')
    ).reset_index()

    X_c = store_agg.drop('Store_ID', axis=1)
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X_c)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(Xs)

    sil = silhouette_score(Xs, km.labels_)
    store_agg['Cluster'] = km.labels_

    with open(MODELS_PATH + "kmeans_clustering.pkl", 'wb') as f:
        pickle.dump(km, f)

    store_agg.to_csv("data/processed/store_clusters.csv", index=False)
    print(f"  ✅ KMeans done  Silhouette={sil:.4f}")
    print(f"  💾 store_clusters.csv saved")


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(reg_metrics, clf_metrics):
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "   🎉 ALL TRAINING COMPLETE — FINAL SUMMARY".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    print("\n  📈 REGRESSION:")
    for name, m in reg_metrics.items():
        print(f"    {name:<25} R²={m['R2']:.4f}  MAE={m['MAE']:,.0f}")

    print("\n  🏷️  CLASSIFICATION:")
    for name, m in clf_metrics.items():
        print(f"    {name:<25} Accuracy={m['Accuracy']*100:.1f}%  F1={m['F1_Score']:.4f}")

    print("\n  📁 Files saved:")
    print("     models/  → 8 pkl files")
    print("     models/results/ → predictions + feature importance + comparison JSON")

    print("\n" + "─"*60)
    print("  🚀 Launch Streamlit Dashboard:")
    print("     streamlit run app.py")
    print("─"*60 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print_banner()

    # Step 1 & 2 check
    check_steps_done()

    # Load & split data
    (X_train, X_test, X_train_sc, X_test_sc,
     y_reg_train, y_reg_test,
     y_clf_train, y_clf_test,
     feature_cols, df, scaler) = load_and_split()

    # Train Regression
    reg_metrics = train_regression(
        X_train, X_test, X_train_sc, X_test_sc,
        y_reg_train, y_reg_test, feature_cols, scaler
    )

    # Train Classification
    clf_metrics = train_classification(
        X_train, X_test, X_train_sc, X_test_sc,
        y_clf_train, y_clf_test, feature_cols, scaler
    )

    # Train Clustering
    train_clustering(df)

    # Save comparison report
    generate_comparison_report(reg_metrics, clf_metrics)

    # Final summary
    final_summary(reg_metrics, clf_metrics)

    elapsed = time.time() - start
    print(f"  ⏱️  Total time: {elapsed:.1f} seconds\n")


if __name__ == "__main__":
    main()
