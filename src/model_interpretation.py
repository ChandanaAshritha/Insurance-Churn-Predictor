import shap
import matplotlib.pyplot as plt
import os

def explain_with_shap(model, X_test, model_name, reports_dir='reports'):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary/multiclass output
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Feature Importance (Bar)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
    plt.title(f'{model_name} - Global Feature Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/shap_plots/{model_name}_feature_importance.png', bbox_inches='tight')
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_vals, X_test, show=False)
    plt.title(f'{model_name} - SHAP Beeswarm Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/shap_plots/{model_name}_beeswarm.png', bbox_inches='tight')
    plt.close()

    return explainer, shap_values