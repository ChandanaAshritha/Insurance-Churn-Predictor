FEATURE_MAPPING = {
    'feature_0': 'premium_to_age_ratio',
    'feature_1': 'claim_frequency',
    'feature_2': 'policy_tenure_scaled',
    'feature_3': 'payment_delay_score',
    'feature_4': 'service_interaction_count',
    'feature_5': 'discount_eligibility_score',
    'feature_6': 'risk_score',
    'feature_7': 'region_code',
    'feature_8': 'sales_channel_id',
    'feature_9': 'policy_type',
    'feature_10': 'renewal_status',
    'feature_11': 'family_plan_flag',
    'feature_12': 'auto_renew_flag',
    'feature_13': 'digital_engagement_level',
    'feature_14': 'days_associated',
    'feature_15': 'claim_count_last_year'
}

# For UI display, we want to map model feature names to business-friendly names.
# The app imports REVERSE_MAPPING and calls REVERSE_MAPPING.get(feature_name, feature_name),
# so this should map internal feature -> display name (same direction as FEATURE_MAPPING).
REVERSE_MAPPING = FEATURE_MAPPING.copy()

__all__ = ["FEATURE_MAPPING", "REVERSE_MAPPING"]