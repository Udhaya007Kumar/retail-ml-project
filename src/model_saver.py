# ============================================================
# MODEL SAVER  💾
# ============================================================
# என்ன பண்றோம்:
#   - Model results (accuracy, R2) save பண்றோம்
#   - Feature importance save பண்றோம்
#   - Predictions save பண்றோம்
#   - Model comparison report generate பண்றோம்
# ============================================================

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

MODELS_PATH  = "models/"
RESULTS_PATH = "models/results/"
os.makedirs(RESULTS_PATH, exist_ok=True)


# ============================================================
# SAVE MODEL WITH METADATA
# ============================================================

def save_model_with_meta(model, model_name, model_type, metrics):
    """
    Model + அதோட results ஒரே dict-ல save பண்றோம்.

    model_type: 'regression' or 'classification' or 'clustering'
    metrics: dict with scores
    """
    # Model save
    model_path = os.path.join(MODELS_PATH, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Metadata save (JSON format)
    meta = {
        "model_name"   : model_name,
        "model_type"   : model_type,
        "trained_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics"      : metrics,
    }

    meta_path = os.path.join(RESULTS_PATH, f"{model_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ Saved: {model_path}")
    print(f"  ✅ Meta : {meta_path}")
    return meta


# ============================================================
# REGRESSION RESULTS SAVE
# ============================================================

def save_regression_results(model, model_name, X_test, y_test, scaler=None):
    """
    Regression model-ன் predictions + metrics save பண்றோம்.
    """
    # Predict
    if scaler and model_name == 'linear_regression':
        X_scaled = scaler.transform(X_test)
        y_pred   = model.predict(X_scaled)
    else:
        y_pred = model.predict(X_test)

    # Metrics calculate
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100)

    metrics = {
        "MAE"  : round(mae, 2),
        "RMSE" : round(rmse, 2),
        "R2"   : round(r2, 4),
        "MAPE" : round(mape, 2),
    }

    # Predictions DataFrame save
    results_df = pd.DataFrame({
        'Actual'   : y_test.values,
        'Predicted': y_pred.round(0),
        'Error'    : (y_test.values - y_pred).round(0)
    })
    results_df.to_csv(
        os.path.join(RESULTS_PATH, f"{model_name}_predictions.csv"),
        index=False
    )

    print(f"\n  📊 {model_name} Regression Results:")
    for k, v in metrics.items():
        print(f"     {k}: {v}")

    return metrics, y_pred


# ============================================================
# CLASSIFICATION RESULTS SAVE
# ============================================================

def save_classification_results(model, model_name, X_test, y_test, scaler=None):
    """
    Classification model-ன் predictions + metrics save பண்றோம்.
    """
    # Predict
    if scaler and model_name == 'logistic_regression':
        X_scaled = scaler.transform(X_test)
        y_pred   = model.predict(X_scaled)
        y_proba  = model.predict_proba(X_scaled)
    else:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Metrics
    acc       = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1        = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    cm        = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "Accuracy"  : round(acc, 4),
        "Precision" : round(precision, 4),
        "Recall"    : round(recall, 4),
        "F1_Score"  : round(f1, 4),
        "Confusion_Matrix": cm
    }

    # Save predictions
    results_df = pd.DataFrame({
        'Actual'   : y_test.values,
        'Predicted': y_pred,
        'Correct'  : (y_test.values == y_pred).astype(int)
    })
    results_df.to_csv(
        os.path.join(RESULTS_PATH, f"{model_name}_predictions.csv"),
        index=False
    )

    print(f"\n  📊 {model_name} Classification Results:")
    print(f"     Accuracy  : {acc*100:.2f}%")
    print(f"     Precision : {precision:.4f}")
    print(f"     Recall    : {recall:.4f}")
    print(f"     F1 Score  : {f1:.4f}")

    return metrics, y_pred


# ============================================================
# FEATURE IMPORTANCE SAVE
# ============================================================

def save_feature_importance(model, feature_cols, model_name):
    """
    Random Forest / XGBoost feature importance save பண்றோம்.
    எந்த feature மிகவும் important-ன்னு தெரியும்.
    """
    if not hasattr(model, 'feature_importances_'):
        return None

    importance_df = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    save_path = os.path.join(RESULTS_PATH, f"{model_name}_feature_importance.csv")
    importance_df.to_csv(save_path, index=False)

    print(f"\n  🔍 Top 5 Important Features ({model_name}):")
    for _, row in importance_df.head(5).iterrows():
        bar = "█" * int(row['Importance'] * 50)
        print(f"     {row['Feature']:30s} {bar} {row['Importance']:.4f}")

    return importance_df


# ============================================================
# MODEL COMPARISON REPORT
# ============================================================

def generate_comparison_report(reg_metrics, clf_metrics):
    """
    எல்லா models-ஐயும் compare பண்ணி report save பண்றோம்.
    """
    print("\n" + "=" * 60)
    print("📋 MODEL COMPARISON REPORT")
    print("=" * 60)

    # Regression comparison
    print("\n  📈 REGRESSION MODELS:")
    print(f"  {'Model':<25} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
    print("  " + "-" * 55)
    for name, m in reg_metrics.items():
        print(f"  {name:<25} {m['R2']:>8.4f} {m['MAE']:>10,.0f} {m['RMSE']:>10,.0f}")

    # Classification comparison
    print("\n  🏷️  CLASSIFICATION MODELS:")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>8}")
    print("  " + "-" * 45)
    for name, m in clf_metrics.items():
        print(f"  {name:<25} {m['Accuracy']*100:>9.1f}% {m['F1_Score']:>8.4f}")

    # Best models
    best_reg = max(reg_metrics, key=lambda x: reg_metrics[x]['R2'])
    best_clf = max(clf_metrics, key=lambda x: clf_metrics[x]['Accuracy'])

    print(f"\n  🏆 Best Regression    : {best_reg}")
    print(f"  🏆 Best Classification: {best_clf}")

    # Save report as JSON
    report = {
        "generated_at"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regression_results"  : reg_metrics,
        "classification_results": clf_metrics,
        "best_regression"     : best_reg,
        "best_classification" : best_clf
    }

    report_path = os.path.join(RESULTS_PATH, "model_comparison.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  💾 Report saved: {report_path}")

    return report


# ============================================================
# LOAD SAVED RESULTS (Streamlit-ல use பண்ண)
# ============================================================

def load_model_results():
    """
    Save பண்ணிய JSON results படிக்கிறோம்.
    Streamlit pages-ல show பண்ண use பண்ணலாம்.
    """
    report_path = os.path.join(RESULTS_PATH, "model_comparison.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return json.load(f)
    return None


def load_predictions(model_name):
    """Saved predictions படிக்கிறோம்."""
    path = os.path.join(RESULTS_PATH, f"{model_name}_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_feature_importance(model_name):
    """Saved feature importance படிக்கிறோம்."""
    path = os.path.join(RESULTS_PATH, f"{model_name}_feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
