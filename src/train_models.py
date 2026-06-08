# ============================================================
# STEP 3: MODEL TRAINING  🤖
# ============================================================
# என்ன பண்றோம்:
#   MODEL 1 - REGRESSION (Sales amount predict)
#     ├── Linear Regression
#     ├── Random Forest Regressor
#     └── XGBoost Regressor
#
#   MODEL 2 - CLASSIFICATION (High/Medium/Low predict)
#     ├── Logistic Regression
#     ├── Random Forest Classifier
#     └── XGBoost Classifier
#
#   MODEL 3 - CLUSTERING (Store/Product grouping)
#     └── KMeans Clustering
#
#   எல்லா models-ஐயும் save பண்றோம் → models/ folder
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle           # Model save/load பண்ண
import warnings
warnings.filterwarnings('ignore')

# ── Scikit-learn ──────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster         import KMeans
from sklearn.metrics         import (
    mean_absolute_error, mean_squared_error, r2_score,   # Regression metrics
    accuracy_score, classification_report,                # Classification metrics
    silhouette_score                                      # Clustering metric
)

# ── XGBoost ───────────────────────────────────────────────
from xgboost import XGBRegressor, XGBClassifier

# ============================================================
# PART 1: PATHS & FOLDERS
# ============================================================

DATA_PATH   = "data/processed/final_dataset.csv"
MODELS_PATH = "models/"
os.makedirs(MODELS_PATH, exist_ok=True)   # models/ folder இல்லன்னா create

print("=" * 60)
print("🤖 MODEL TRAINING STARTED")
print("=" * 60)


# ============================================================
# PART 2: DATA LOAD & FEATURES SELECT
# ============================================================

def load_and_prepare(filepath):
    """
    Final dataset படிக்கிறோம்.
    ML-க்கு தேவையான feature columns select பண்றோம்.
    """
    print("\n📂 Loading final dataset...")
    df = pd.read_csv(filepath)
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Features (X) ─────────────────────────────────────
    # இந்த columns வச்சு predict பண்றோம்
    feature_cols = [
        'Month', 'DayOfWeek', 'Quarter', 'Is_Weekend', 'Is_Holiday',
        'Season_Encoded', 'Category_Encoded', 'Brand_Encoded',
        'Store_Type_Encoded', 'Region_Encoded', 'Price_Range_Encoded',
        'Payment_Method_Encoded', 'Has_Promotion', 'Max_Discount',
        'Store_Age_Years', 'Total_Stock', 'Customer_Count',
        'Profit_Margin', 'Revenue_Per_Customer', 'Avg_Discount',
        'Stock_Available', 'Discount_Sales_Ratio'
    ]

    # DataFrame-ல இருக்கற columns மட்டும் எடுக்கிறோம்
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]                           # Features
    y_reg   = df['Sales']                          # Regression target
    y_class = df['Sales_Category_Encoded']         # Classification target

    # NaN values → 0 replace
    X = X.fillna(0)
    y_reg   = y_reg.fillna(y_reg.median())
    y_class = y_class.fillna(1)

    print(f"   Features selected : {len(feature_cols)}")
    print(f"   Feature columns   : {feature_cols}")

    return X, y_reg, y_class, df


# ============================================================
# PART 3: TRAIN-TEST SPLIT
# ============================================================
# Data-ஐ 2 பகுதியா பிரிக்கிறோம்:
#   Training data (80%) → Model இதை பாத்து கத்துக்கும்
#   Testing data  (20%) → Model இதை பாத்தே இல்லை → accuracy check
#
# random_state=42 → every time same split வரும் (reproducible)
# ============================================================

def split_data(X, y_reg, y_class):
    print("\n✂️  Splitting data (80% train / 20% test)...")

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    _, _, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    print(f"   Training rows : {len(X_train):,}")
    print(f"   Testing rows  : {len(X_test):,}")

    return X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test


# ============================================================
# PART 4: SCALING (Normalize பண்றோம்)
# ============================================================
# Features-ல values மிகவும் different-ஆ இருக்கும்:
#   Customer_Count: 1 - 100
#   Sales: 500 - 9999
#   Total_Stock: 0 - 10000
#
# StandardScaler → எல்லாத்தையும் same scale-ஆ மாத்தும்
# Linear Regression, Logistic Regression-க்கு இது மிகவும் முக்கியம்
# ============================================================

def scale_features(X_train, X_test):
    print("\n📏 Scaling features...")

    scaler = StandardScaler()

    # fit_transform → train data-ல scale கத்துக்கும் + apply பண்ணும்
    X_train_scaled = scaler.fit_transform(X_train)

    # transform only → test data-ல apply மட்டும் பண்ணும் (கத்துக்காது)
    X_test_scaled  = scaler.transform(X_test)

    print("   ✅ Scaling done")
    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# PART 5: MODEL SAVE பண்றோம்
# ============================================================

def save_model(model, filename):
    """
    pickle → Python object-ஐ file-ஆ save பண்றது
    Later load பண்ணி predict பண்ணலாம்
    """
    path = MODELS_PATH + filename
    with open(path, 'wb') as f:   # 'wb' = write binary
        pickle.dump(model, f)
    print(f"   💾 Saved: {path}")


# ============================================================
# PART 6: REGRESSION MODELS
# ============================================================
# Regression = Number predict பண்றது
# Target: Sales amount (500 to 9999)
# ============================================================

def train_regression_models(X_train_sc, X_test_sc, X_train, X_test,
                             y_train, y_test):
    print("\n" + "=" * 60)
    print("📈 REGRESSION MODELS (Predict Sales Amount)")
    print("=" * 60)

    results = {}

    # ── Model 1A: Linear Regression ──────────────────────
    # Simple straight line relationship find பண்றது
    # y = m1*x1 + m2*x2 + ... + b
    print("\n  🔹 Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)       # Train!
    y_pred = lr.predict(X_test_sc)    # Predict!

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"     MAE (Mean Abs Error) : {mae:,.0f}")
    print(f"     R² Score             : {r2:.4f}  (1.0 = perfect)")
    results['Linear Regression'] = {'MAE': mae, 'R2': r2}
    save_model(lr, "linear_regression.pkl")

    # ── Model 1B: Random Forest Regressor ────────────────
    # 100 decision trees உருவாக்கும்
    # எல்லா trees-ம் predict பண்ணும் → average எடுக்கும்
    # n_estimators = எத்தனை trees
    print("\n  🔹 Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=100,    # 100 trees
        max_depth=10,        # Tree எவ்வளவு deep போகலாம்
        random_state=42,
        n_jobs=-1            # எல்லா CPU cores use பண்ணு
    )
    rf.fit(X_train, y_train)          # Scaled இல்லாத data use
    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"     MAE (Mean Abs Error) : {mae:,.0f}")
    print(f"     R² Score             : {r2:.4f}")
    results['Random Forest'] = {'MAE': mae, 'R2': r2}
    save_model(rf, "rf_regressor.pkl")

    # ── Model 1C: XGBoost Regressor ──────────────────────
    # Gradient Boosting — ஒவ்வொரு tree முந்தையதோட mistakes fix பண்ணும்
    # மிகவும் powerful, competitions-ல winning algorithm!
    print("\n  🔹 XGBoost Regressor...")
    xgb = XGBRegressor(
        n_estimators=200,    # 200 trees
        learning_rate=0.1,   # எவ்வளவு fast கத்துக்கணும்
        max_depth=6,
        random_state=42,
        verbosity=0          # Progress messages hide
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"     MAE (Mean Abs Error) : {mae:,.0f}")
    print(f"     R² Score             : {r2:.4f}")
    results['XGBoost'] = {'MAE': mae, 'R2': r2}
    save_model(xgb, "xgb_regressor.pkl")

    # ── Best Model Print ──────────────────────────────────
    best = max(results, key=lambda x: results[x]['R2'])
    print(f"\n  🏆 Best Regression Model: {best} (R²={results[best]['R2']:.4f})")

    return results


# ============================================================
# PART 7: CLASSIFICATION MODELS
# ============================================================
# Classification = Category predict பண்றது
# Target: Low(0), Medium(1), High(2)
# ============================================================

def train_classification_models(X_train_sc, X_test_sc, X_train, X_test,
                                  y_train, y_test):
    print("\n" + "=" * 60)
    print("🏷️  CLASSIFICATION MODELS (Predict Low/Medium/High)")
    print("=" * 60)

    results = {}

    # ── Model 2A: Logistic Regression ────────────────────
    # Regression மாதிரி இருக்கும் ஆனா category predict பண்ணும்
    # Probability calculate பண்ணி class decide பண்ணும்
    print("\n  🔹 Logistic Regression...")
    log_reg = LogisticRegression(
        max_iter=1000,       # 1000 iterations வரை try பண்ணும்
        random_state=42
        # 3 classes automatically handle ஆகும்
    )
    log_reg.fit(X_train_sc, y_train)
    y_pred = log_reg.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    print(f"     Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    results['Logistic Regression'] = acc
    save_model(log_reg, "logistic_regression.pkl")

    # ── Model 2B: Random Forest Classifier ───────────────
    print("\n  🔹 Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"     Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print("\n     Detailed Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=['Low', 'Medium', 'High'],
        zero_division=0
    )
    for line in report.split('\n'):
        if line.strip():
            print(f"       {line}")
    results['Random Forest'] = acc
    save_model(rf_clf, "rf_classifier.pkl")

    # ── Model 2C: XGBoost Classifier ─────────────────────
    print("\n  🔹 XGBoost Classifier...")
    xgb_clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0,
        eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"     Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    results['XGBoost'] = acc
    save_model(xgb_clf, "xgb_classifier.pkl")

    # ── Best Model ────────────────────────────────────────
    best = max(results, key=results.get)
    print(f"\n  🏆 Best Classification Model: {best} ({results[best]*100:.1f}%)")

    return results


# ============================================================
# PART 8: CLUSTERING MODEL
# ============================================================
# Clustering = Groups உருவாக்குறது (no target needed!)
# Store performance-ஐ பாத்து groups பண்றோம்
# ============================================================

def train_clustering_model(df):
    print("\n" + "=" * 60)
    print("🔵 CLUSTERING MODEL (Find Store Groups)")
    print("=" * 60)

    # Store-level aggregation பண்றோம்
    store_features = df.groupby('Store_ID').agg(
        Total_Sales    = ('Sales', 'sum'),
        Avg_Sales      = ('Sales', 'mean'),
        Total_Customers= ('Customer_Count', 'sum'),
        Avg_Quantity   = ('Quantity', 'mean'),
        Transaction_Count = ('Sale_ID', 'count')
    ).reset_index()

    print(f"\n   Store features shape: {store_features.shape}")

    # Scale features
    X_cluster = store_features.drop('Store_ID', axis=1)
    scaler_c  = StandardScaler()
    X_scaled  = scaler_c.fit_transform(X_cluster)

    # ── KMeans Clustering ─────────────────────────────────
    # n_clusters=3 → 3 groups: High, Medium, Low performing stores
    print("\n  🔹 KMeans Clustering (k=3)...")
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10        # 10 different starts → best one எடுக்கும்
    )
    kmeans.fit(X_scaled)

    # Silhouette Score → clusters எவ்வளவு நல்லா separated?
    # Score: -1 to 1, higher = better
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"     Silhouette Score : {sil_score:.4f}  (higher is better)")

    # Cluster labels add பண்றோம்
    store_features['Cluster'] = kmeans.labels_

    # Cluster summary
    print("\n     Cluster Summary:")
    summary = store_features.groupby('Cluster').agg(
        Stores=('Store_ID', 'count'),
        Avg_Sales=('Avg_Sales', 'mean'),
        Avg_Customers=('Avg_Sales', 'mean')
    ).round(0)
    print(summary.to_string())

    save_model(kmeans, "kmeans_clustering.pkl")
    store_features.to_csv("data/processed/store_clusters.csv", index=False)
    print(f"\n   💾 Store clusters saved: data/processed/store_clusters.csv")

    return kmeans, sil_score


# ============================================================
# PART 9: SCALER SAVE பண்றோம்
# ============================================================

def save_scaler(scaler):
    """
    Scaler-ஐயும் save பண்றோம்.
    Streamlit-ல predict பண்ணும்போது same scaler use பண்ணணும்!
    """
    save_model(scaler, "scaler.pkl")
    print("   ✅ Scaler saved")


# ============================================================
# PART 10: FINAL SUMMARY
# ============================================================

def print_final_summary(reg_results, clf_results, sil_score):
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE — FINAL RESULTS")
    print("=" * 60)

    print("\n  📈 REGRESSION (Sales Predict):")
    for model, scores in reg_results.items():
        print(f"     {model:25s} → R²={scores['R2']:.4f}, MAE={scores['MAE']:,.0f}")

    print("\n  🏷️  CLASSIFICATION (High/Med/Low):")
    for model, acc in clf_results.items():
        print(f"     {model:25s} → Accuracy={acc*100:.1f}%")

    print(f"\n  🔵 CLUSTERING:")
    print(f"     KMeans (k=3)              → Silhouette={sil_score:.4f}")

    print("\n  📁 Models saved in: models/")
    print("     ✅ linear_regression.pkl")
    print("     ✅ rf_regressor.pkl")
    print("     ✅ xgb_regressor.pkl")
    print("     ✅ logistic_regression.pkl")
    print("     ✅ rf_classifier.pkl")
    print("     ✅ xgb_classifier.pkl")
    print("     ✅ kmeans_clustering.pkl")
    print("     ✅ scaler.pkl")

    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("   Next step → streamlit run app.py")
    print("=" * 60)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # Step 1: Load data
    X, y_reg, y_class, df = load_and_prepare(DATA_PATH)

    # Step 2: Train/Test split
    X_train, X_test, y_reg_train, y_reg_test, \
    y_class_train, y_class_test = split_data(X, y_reg, y_class)

    # Step 3: Scale features
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # Step 4: Train Regression Models
    reg_results = train_regression_models(
        X_train_sc, X_test_sc,
        X_train, X_test,
        y_reg_train, y_reg_test
    )

    # Step 5: Train Classification Models
    clf_results = train_classification_models(
        X_train_sc, X_test_sc,
        X_train, X_test,
        y_class_train, y_class_test
    )

    # Step 6: Train Clustering
    kmeans, sil_score = train_clustering_model(df)

    # Step 7: Save scaler
    save_scaler(scaler)

    # Step 8: Final summary
    print_final_summary(reg_results, clf_results, sil_score)


if __name__ == "__main__":
    main()
