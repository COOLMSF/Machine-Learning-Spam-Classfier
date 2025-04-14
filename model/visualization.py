import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def create_visualization_directory():
    """Create directory for saving visualization plots"""
    viz_dir = "visualization_results"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir

def plot_model_comparison(accuracies, times, viz_dir):
    """Plot model comparison of accuracy and execution time"""
    models = ['Random Forest', 'SVM', 'Naive Bayes']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(models, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax1.set_ylabel('Accuracy Score')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Execution time comparison
    bars2 = ax2.bar(models, times, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Model Execution Time Comparison')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/model_comparison.png")
    plt.close()

def plot_confusion_matrices(y_test, rf_pred, svm_pred, nb_pred, viz_dir):
    """Plot confusion matrices for all models"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot confusion matrix for Random Forest
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Random Forest\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot confusion matrix for SVM
    cm_svm = confusion_matrix(y_test, svm_pred)
    sns.heatmap(cm_svm, annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title('SVM\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # Plot confusion matrix for Naive Bayes
    cm_nb = confusion_matrix(y_test, nb_pred)
    sns.heatmap(cm_nb, annot=True, fmt='d', ax=ax3, cmap='Blues')
    ax3.set_title('Naive Bayes\nConfusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/confusion_matrices.png")
    plt.close()

def plot_roc_curves(y_test, rf_proba, svm_proba, nb_proba, viz_dir):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba[:, 1])
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    # Calculate ROC curve and AUC for SVM
    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_proba[:, 1])
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    # Calculate ROC curve and AUC for Naive Bayes
    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_proba[:, 1])
    roc_auc_nb = auc(fpr_nb, tpr_nb)
    
    # Plot all ROC curves
    plt.plot(fpr_rf, tpr_rf, color='#2ecc71', label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
    plt.plot(fpr_svm, tpr_svm, color='#3498db', label=f'SVM (AUC = {roc_auc_svm:.4f})')
    plt.plot(fpr_nb, tpr_nb, color='#e74c3c', label=f'Naive Bayes (AUC = {roc_auc_nb:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/roc_curves.png")
    plt.close()

def plot_feature_importance(vectorizer, rf_classifier, viz_dir, top_n=20):
    """Plot feature importance from Random Forest classifier"""
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_classifier.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.title(f'Top {top_n} Most Important Features')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/feature_importance.png")
    plt.close()

def plot_learning_curves(model_name, train_sizes, train_scores, test_scores, viz_dir):
    """Plot learning curves for a model"""
    plt.figure(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, label='Training score', color='#2ecc71')
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='#3498db')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#2ecc71')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='#3498db')
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves for {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/learning_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.close()