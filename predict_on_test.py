# predict_on_test.py — UPDATED

import pandas as pd
import joblib
from src.data_preprocessing import load_and_clean_data, encode_categorical_features, handle_outliers
import os

def main():
    print("🔮 Loading trained model...")
    model = joblib.load('models/best_model.pkl')

    print("📂 Loading test data...")
    df_test = load_and_clean_data('data/Test.csv')

    # Clean — no encoding needed
    df_test = encode_categorical_features(df_test)

    # Handle outliers (same as train)
    feature_cols = [col for col in df_test.columns if col.startswith('feature_')]
    df_test = handle_outliers(df_test, feature_cols)

    # Align features with model
    feature_names = getattr(model, 'feature_name_', None)
    if feature_names is None:
        feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is not None:
        for col in feature_names:
            if col not in df_test.columns:
                df_test[col] = 0
        df_test = df_test[list(feature_names)]

    # Predict
    print("📈 Predicting churn probabilities...")
    y_proba = model.predict_proba(df_test)[:, 1]

    # Format submission — match sample_submission.xlsx structure
    submission = pd.DataFrame({
        'labels': y_proba  # sample_submission uses 'labels' column
    })

    # Save to output
    os.makedirs('output', exist_ok=True)
    submission.to_csv('output/submission.csv', index=False)
    print(f"✅ Submission saved: {len(submission)} rows")
    print(submission.head())

    # Validate format
    if os.path.exists('data/sample_submission.csv'):
        sample = pd.read_csv('data/sample_submission.csv')
        assert len(submission) == len(sample), "❌ Length mismatch with sample_submission.csv"
        print("✅ Format validated against sample_submission.csv")

if __name__ == "__main__":
    main()