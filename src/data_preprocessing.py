# src/data_preprocessing.py

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from .feature_mapping import FEATURE_MAPPING

def load_and_clean_data(filepath):
    """
    Load and clean dataset — fill any missing values.
    Then rename columns using FEATURE_MAPPING for business clarity.
    """
    df = pd.read_csv(filepath)
    
    # Fill missing values with median (safe for numerical data)
    df = df.fillna(df.median(numeric_only=True))
    
    # Rename features to business-friendly names (safe — original files unchanged)
    df = df.rename(columns=FEATURE_MAPPING)
    
    return df

def encode_categorical_features(df):
    """
    NO CATEGORICAL FEATURES TO ENCODE.
    Your data is already numerical — this function does nothing.
    """
    return df

def handle_outliers(df, columns):
    """
    Optional: Cap outliers using IQR for numerical stability.
    Applied only if column exists.
    """
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

def prepare_train_test(df, target_col='labels', test_size=0.2, random_state=42):
    """
    Prepare train/val split.
    Uses 'labels' as target — matches your Train.csv.
    """
    if target_col not in df.columns:
        target_col = df.columns[-1]  # Fallback to last column

    X = df.drop([target_col], axis=1, errors='ignore')
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_val, y_train_res, y_val