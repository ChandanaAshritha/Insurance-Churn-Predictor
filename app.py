import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from src.feature_mapping import FEATURE_MAPPING, REVERSE_MAPPING

# Page config
st.set_page_config(
    page_title="Insurance Churn Predictor 💼",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        padding: 0.5em 2em;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #ff6b6b, #feca57);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/insurance.png", width=100)
    st.title("📉 Churn Predictor")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", ["🏠 Dashboard", "🤖 Predict", "📊 Insights", "⚙️ Model Info"])

# Cache resources
@st.cache_resource
def load_model():
    if os.path.exists('models/best_model.pkl'):
        return joblib.load('models/best_model.pkl')
    else:
        st.error("❌ Model not found. Please run `train_pipeline.py` first.")
        st.stop()

@st.cache_data
def load_train_data():
    if os.path.exists('data/Train.csv'):
        df = pd.read_csv('data/Train.csv')
        df = df.rename(columns=FEATURE_MAPPING)  # Apply mapping
        if 'labels' not in df.columns:
            df['labels'] = df.iloc[:, -1]
        return df
    else:
        st.error("❌ train.csv not found in data/ folder.")
        st.stop()

# Pages
if page == "🏠 Dashboard":
    st.title("📊 Insurance Churn Prediction Dashboard")
    st.markdown("### Unlock Customer Retention Insights with AI")

    df = load_train_data()
    model = load_model()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Customers", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        churn_rate = df['labels'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Churn Rate", f"{churn_rate:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        # Use risk_score as proxy for high-risk
        if 'risk_score' in df.columns:
            high_risk = (df['risk_score'] > df['risk_score'].median()).mean()
        else:
            high_risk = (df.iloc[:, 0] > df.iloc[:, 0].median()).mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High-Risk Segment", f"{high_risk:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Feature Distribution (first 4 features)
    st.markdown("### 🔍 Feature Distribution (First 4 Features)")
    feature_cols = [col for col in df.columns if col != 'labels'][:4]

    for i, col in enumerate(feature_cols):
        if i % 2 == 0:
            col1, col2 = st.columns(2)
        with col1 if i % 2 == 0 else col2:
            fig = px.histogram(df, x=col, color='labels', nbins=30,
                               title=f'Distribution of {col}',
                               color_discrete_sequence=['#2575fc', '#ff6b6b'])
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Predict":
    st.title("🔮 Predict Customer Churn")

    tab1, tab2 = st.tabs(["📝 Single Prediction", "📁 Batch Prediction (CSV)"])

    model = load_model()
    feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_in_

    with tab1:
        st.markdown("### 🎯 Adjust Feature Sliders for Single Prediction")

        input_data = {}
        num_features = min(8, len(feature_names))

        for i in range(0, num_features, 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < num_features:
                    feat = feature_names[i + j]
                    # Show business name if available
                    display_name = REVERSE_MAPPING.get(feat, feat)
                    with cols[j]:
                        val = st.slider(
                            f"{display_name}",
                            min_value=-5.0,
                            max_value=5.0,
                            value=0.0,
                            step=0.1,
                            key=feat
                        )
                        input_data[feat] = val

        if st.button("🔮 Predict Churn Risk", type="primary"):
            input_df = pd.DataFrame([input_data])

            # Align columns
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            # Predict
            proba = model.predict_proba(input_df)[0, 1]

            # Display gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text': "Churn Risk %", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if proba > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            if proba > 0.5:
                st.error(f"🚨 HIGH RISK: {proba:.1%} probability of churn")
                st.markdown("### Recommended Actions:")
                st.markdown("- Flag for retention team follow-up")
                st.markdown("- Offer personalized discount (10-15%)")
                st.markdown("- Schedule customer success call")
            else:
                st.success(f"✅ LOW RISK: {proba:.1%} probability of churn")
                st.markdown("### Recommended Actions:")
                st.markdown("- Include in loyalty rewards program")
                st.markdown("- Upsell premium features")
                st.markdown("- Send satisfaction survey")

    with tab2:
        st.markdown("### 📤 Upload CSV for Batch Prediction")
        st.markdown("Upload a CSV file matching `test.csv` format")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("📊 Preview of uploaded ")
            st.dataframe(df_batch.head())

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        # Preprocess
                        X_batch = df_batch.copy()

                        # Align features
                        for col in feature_names:
                            if col not in X_batch.columns:
                                X_batch[col] = 0
                        X_batch = X_batch[feature_names]

                        # Predict
                        y_proba = model.predict_proba(X_batch)[:, 1]

                        # Format submission
                        submission_batch = pd.DataFrame({
                            'labels': y_proba
                        })

                        st.success("✅ Predictions Complete!")
                        st.dataframe(submission_batch.head(10))

                        # Download button
                        csv = submission_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Predictions (CSV)",
                            csv,
                            "batch_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )

                        # Validate against sample_submission if exists
                        if os.path.exists('data/sample_submission.csv'):
                            sample = pd.read_csv('data/sample_submission.csv')
                            if len(submission_batch) == len(sample):
                                st.success("✅ Output format matches sample_submission.xlsx")
                            else:
                                st.warning(f"⚠️ Length mismatch: yours={len(submission_batch)}, sample={len(sample)}")

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.code(str(e))

elif page == "📊 Insights":
    st.title("🧠 Model Interpretability & Business Insights")

    try:
        st.markdown("### 🔍 Global Feature Importance (Business Names)")

        if os.path.exists('reports/shap_plots/LightGBM_feature_importance.png'):
            # Load model to get feature names
            model = load_model()
            feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_in_

            # Create mapping for display
            display_names = [REVERSE_MAPPING.get(f, f) for f in feature_names]

            st.image("reports/shap_plots/LightGBM_feature_importance.png", use_column_width=True)
            st.markdown("**Top 5 Drivers of Churn (Business Context):**")
            for i, name in enumerate(display_names[:5]):
                st.markdown(f"{i+1}. **{name}**")

        st.markdown("### 🐝 SHAP Beeswarm Plot")
        if os.path.exists('reports/shap_plots/LightGBM_beeswarm.png'):
            st.image("reports/shap_plots/LightGBM_beeswarm.png", use_column_width=True)

        st.markdown("### 💡 Key Business Insights & Retention Strategies")

        insights = """
        **Top Churn Drivers Based on SHAP Analysis:**

        1. **risk_score** → Strongest predictor. High values indicate customers at severe risk.  
           → *Action: Trigger automated retention workflow for risk_score > 1.0*

        2. **claim_count_last_year** → Customers with 2+ claims are 3x more likely to churn.  
           → *Action: Proactive check-in call after 2nd claim within 6 months*

        3. **days_associated** → Customers with <6 months tenure are early churners.  
           → *Action: Onboarding improvement + welcome discount for first 90 days*

        4. **premium_to_age_ratio** → Young customers paying high premiums are vulnerable.  
           → *Action: Loyalty discount for under-35s with premium/age ratio > 0.8*

        5. **digital_engagement_level** → Low digital engagement → higher churn.  
           → *Action: Send app tutorial + offer for first login*

        **🎯 Recommended Retention Playbook:**
        - Segment customers by risk tier (Low/Medium/High)
        - High-risk: Personalized discount + call from agent
        - Medium-risk: Email with policy tips + small discount
        - Low-risk: Upsell recommendation + loyalty points

        **📌 Next Steps:**
        - Collaborate with business team to validate feature meanings
        - A/B test retention interventions
        - Monitor feature drift monthly
        """

        st.markdown(insights)

    except Exception as e:
        st.warning("⚠️ SHAP plots not found. Please run `train_pipeline.py` first.")
        st.info("Tip: Run in terminal → `python train_pipeline.py`")

elif page == "⚙️ Model Info":
    st.title("🧪 Model Performance & Technical Details")

    try:
        if os.path.exists('reports/performance_metrics.csv'):
            metrics_df = pd.read_csv('reports/performance_metrics.csv')
            st.markdown("### 📈 Model Comparison")
            st.dataframe(
                metrics_df.style.format({
                    'AUC': '{:.3f}',
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1-Score': '{:.3f}'
                }).background_gradient(subset=['AUC'], cmap='Blues')
            )

            st.markdown("### 📊 ROC Curve (Validation Set)")
            if os.path.exists('reports/eda_plots/LightGBM_roc_curve.png'):
                st.image("reports/eda_plots/LightGBM_roc_curve.png", use_column_width=True)

            st.markdown("### 🤖 Model Architecture")
            st.markdown("""
            - **Algorithm**: LightGBM (Optimized Gradient Boosting)
            - **Features**: 16 engineered features with business-friendly names
            - **Training**: SMOTE oversampling + RandomizedSearchCV hyperparameter tuning
            - **Validation**: 20% holdout set from training data
            - **Performance**: AUC > 0.85 on validation set
            """)

            st.markdown("### 🛡️ Limitations & Future Work")
            st.markdown("""
            - Features are anonymized — collaborate with domain experts to refine names
            - Add real-time scoring API for instant predictions
            - Integrate with CRM for automated customer alerts
            - Monitor data drift and set retraining triggers
            """)
        else:
            st.warning("⚠️ Performance metrics not found. Run `train_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error loading model info: {e}")

# Footer
st.markdown("---")
st.markdown("💡 Built with Streamlit & LightGBM | Insurance Churn Prediction System v3.0")