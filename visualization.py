import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

def plot_training_curves(fold_metrics, save_path=None):
    
    num_folds = len(fold_metrics)
    
    fig, axes = plt.subplots(1, num_folds, figsize=(6*num_folds, 5), sharey=True)
    if num_folds == 1:
        axes = [axes]
    for i, metrics in enumerate(fold_metrics):
        ax = axes[i]
        ax.plot(metrics['train_loss'], label='Train Loss', alpha=0.7)
        ax.plot(metrics['val_loss'], label='Val Loss', linestyle='--', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Fold {i+1}')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Training and Validation Loss per Fold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        loss_save_path = save_path.replace('.png', '_loss.png')
        fig.savefig(loss_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    # Plot accuracy curves
    fig, axes = plt.subplots(1, num_folds, figsize=(6*num_folds, 5), sharey=True)
    if num_folds == 1:
        axes = [axes]
    for i, metrics in enumerate(fold_metrics):
        ax = axes[i]
        ax.plot(metrics['train_acc'], label='Train Acc', alpha=0.7)
        ax.plot(metrics['val_acc'], label='Val Acc', linestyle='--', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Fold {i+1}')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Training and Validation Accuracy per Fold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        acc_save_path = save_path.replace('.png', '_accuracy.png')
        fig.savefig(acc_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_ensemble_performance(ensemble_metrics, save_path=None):
    
    plt.figure(figsize=(8, 8))  
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(ensemble_metrics['train_loss'], label='Train')
    plt.plot(ensemble_metrics['val_loss'], label='Validation')
    plt.title('Ensemble Model Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(ensemble_metrics['train_acc'], label='Train')
    plt.plot(ensemble_metrics['val_acc'], label='Validation')
    plt.title('Ensemble Model Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)  
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path=None):
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(importance_scores, feature_names, save_path=None):
   
    plt.figure(figsize=(8, 8))  
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Analysis')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=400)  
    plt.close() 