import pandas as pd

def create_ratio_features(df):
    required = ['premium_amount', 'age', 'days_associated', 'claim_count']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    df['premium_to_age_ratio'] = df['premium_amount'] / (df['age'] + 1)
    df['days_to_age_ratio'] = df['days_associated'] / (df['age'] + 1)
    df['claim_frequency'] = df['claim_count'] / ((df['days_associated'] / 365) + 1)
    return df

def create_interaction_features(df):
    required = ['age', 'premium_amount', 'claim_count']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    df['age_x_premium'] = df['age'] * df['premium_amount']
    df['premium_x_claims'] = df['premium_amount'] * (df['claim_count'] + 1)
    return df

def engineer_features(df):
    df = create_ratio_features(df)
    df = create_interaction_features(df)
    return df