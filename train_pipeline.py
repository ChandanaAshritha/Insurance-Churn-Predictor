# train_pipeline.py — UPDATED

from src.utils import setup_project_directories
from src.data_preprocessing import load_and_clean_data, encode_categorical_features, handle_outliers, prepare_train_test
from src.model_training import train_baseline_model, train_advanced_models, save_model
from src.model_evaluation import evaluate_model
from src.model_interpretation import explain_with_shap
import pandas as pd

def main():
    print("🚀 Starting Churn Prediction Training Pipeline...")

    setup_project_directories()

    # Load and preprocess TRAIN data
    print("📂 Loading training data...")
    df = load_and_clean_data('data/Train.csv')

    # No categorical encoding needed
    df = encode_categorical_features(df)

    # Optional: handle outliers on numerical features (exclude target)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'labels' in numeric_cols:
        numeric_cols.remove('labels')
    df = handle_outliers(df, numeric_cols)

    # Split train/val
    X_train_res, X_val, y_train_res, y_val = prepare_train_test(df, target_col='labels')

    # Train models
    print("🧠 Training models...")
    baseline_model = train_baseline_model(X_train_res, y_train_res)
    advanced_models = train_advanced_models(X_train_res, y_train_res)

    # Evaluate
    print("📊 Evaluating models...")
    all_metrics = []

    baseline_metrics = evaluate_model(baseline_model, X_val, y_val, "Logistic Regression")
    all_metrics.append(baseline_metrics)

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in advanced_models.items():
        metrics = evaluate_model(model, X_val, y_val, name)
        all_metrics.append(metrics)
        if metrics['AUC'] > best_score:
            best_score = metrics['AUC']
            best_model = model
            best_name = name

    # Save best model
    save_model(best_model, 'models/best_model.pkl')
    print(f"🏆 Best Model: {best_name} (AUC: {best_score:.4f})")

    # SHAP (only for tree-based models)
    if best_name in ['LightGBM', 'Random Forest']:
        explain_with_shap(best_model, X_val, best_name)

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('reports/performance_metrics.csv', index=False)
    print("✅ Metrics saved to reports/performance_metrics.csv")
    print("🎉 Training pipeline completed!")

if __name__ == "__main__":
    main()