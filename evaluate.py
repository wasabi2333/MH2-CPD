import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Evaluator:
    
    
    def __init__(self, model, device, config):
        
        self.model = model.to(device)
        self.device = device
        self.config = config
        
    def evaluate(self, test_loader, H):
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        H = H.to(self.device)
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                
                outputs = self.model(features, H)
                
                
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'predictions': y_pred,
            'labels': y_true,
            'probabilities': y_proba
        }
        
        return results
    
    def print_results(self, results):
        
        print("\n" + "="*60)
        print("Model Evaluation Results")
        print("="*60)
        
        print(f"\n{self.config.label_col} Evaluation Metrics:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  AUC-ROC: {results['auc']:.4f}")
        print("="*60)
    
    def plot_confusion_matrix(self, results, save_path=None):
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Low Risk', 'High Risk'], 
                   yticklabels=['Low Risk', 'High Risk'])
        
        ax.set_title(f'{self.config.label_col} Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, results, save_path=None):
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        y_true = results['labels']
        y_scores = results['probabilities']
        
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = results['auc']
        
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{self.config.label_col} ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, results, save_path=None):
        
        report_text = f"Classification Report for {self.config.label_col}\n"
        report_text += "="*80 + "\n\n"
        
        y_true = results['labels']
        y_pred = results['predictions']
        
        report = classification_report(
            y_true, 
            y_pred,
            labels=[0, 1],
            target_names=['Low Risk', 'High Risk']
        )
        
        report_text += report
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_prediction_distribution(self, results, save_path=None):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        probabilities = results['probabilities']
        labels = results['labels']
        
        
        ax1.hist(probabilities[labels == 0], bins=20, alpha=0.5, label='True Low Risk', color='blue')
        ax1.hist(probabilities[labels == 1], bins=20, alpha=0.5, label='True High Risk', color='red')
        ax1.set_xlabel('Predicted Probability of High Risk')
        ax1.set_ylabel('Count')
        ax1.set_title('Prediction Probability Distribution by True Label')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        
        ax2.hist(probabilities, bins=30, edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        ax2.set_xlabel('Predicted Probability of High Risk')
        ax2.set_ylabel('Count')
        ax2.set_title('Overall Prediction Probability Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()