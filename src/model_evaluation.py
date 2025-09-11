from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name, reports_dir='reports'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    pos_label = str(y_test.max())
    metrics = {
        'Model': model_name,
        'AUC': auc_score,
        'Precision': report.get(pos_label, {}).get('precision', 0),
        'Recall': report.get(pos_label, {}).get('recall', 0),
        'F1-Score': report.get(pos_label, {}).get('f1-score', 0)
    }

    # Confusion Matrix
    plt.figure(figsize=(6,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{reports_dir}/eda_plots/{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', color='#FF6F61', linewidth=3)
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{reports_dir}/eda_plots/{model_name}_roc_curve.png', bbox_inches='tight')
    plt.close()

    return metrics