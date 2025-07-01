import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd


from config import Config
from data_loader import load_data, split_data, normalize_features, create_dataloaders
from hypergraph_builder import HypergraphBuilder
from model import HyperGNN
from model_ensemble import EnsembleModel
from data_augmentation import DataAugmentor
from train import Trainer
from evaluate import Evaluator
from cross_validation import run_cross_validation
from utils import (
    set_seed, create_directories, save_config, plot_training_history,
    visualize_hypergraph_statistics, save_experiment_results, print_model_summary
)
from visualization import plot_training_curves, plot_ensemble_performance, plot_confusion_matrix

def train_ensemble():
    config = Config()
    
    
    plots_dir = os.path.join('plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        
        config_save_path = os.path.join('config.json')
        config_dict = {k: str(v) if isinstance(v, torch.device) else v 
                      for k, v in config.__dict__.items()}
        with open(config_save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
        
        print("\n[1/6] Loading data...")
        X, y = load_data(config.data_path, config.feature_cols, config.label_col)
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, config.test_size, config.random_state
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        
        X_train, X_test, scaler = normalize_features(X_train, X_test)
            
        
        os.makedirs('models', exist_ok=True)
        np.save('models/feature_scaler.npy', scaler)
        
        
        train_loader, test_loader = create_dataloaders(
            X_train, X_test, y_train, y_test, config.batch_size
        )
        
        
        print("\n[2/6] Building hypergraph...")
        hypergraph_builder = HypergraphBuilder(config.feature_cols, config.factor_groups)
        
        
        H, hyperedge_info = hypergraph_builder.build_hypergraph(
            X_train, 
            mi_threshold=config.mi_threshold,
            n_clusters=config.n_clusters
        )
            
        
        H = torch.tensor(H, dtype=torch.float32).to(config.device)
        
        
        hypergraph_stats = hypergraph_builder.get_hypergraph_statistics(H.cpu().numpy(), hyperedge_info)
        
        
        stats_plot_path = os.path.join(plots_dir, 'hypergraph_stats.png')
        visualize_hypergraph_statistics(hypergraph_stats, stats_plot_path)
        print(f"Saved hypergraph statistics plot to: {stats_plot_path}")
        
        
        fold_data = []  
        fold_metrics = []  
        
        
        best_model_path = None
        best_accuracy = 0
        
        for fold in range(config.num_folds):
            print(f"\nTraining Fold {fold + 1}")
            
            
            model = HyperGNN(
                input_dim=len(config.feature_cols),
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_classes=config.num_classes
            ).to(config.device)
            
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            
            fold_metric = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            
            for epoch in range(config.num_epochs):
                model.train()
                epoch_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(config.device), targets.to(config.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs, H)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    if (batch_idx + 1) % 10 == 0:
                        
                        _, predicted = outputs.max(1)
                        y_true = targets.cpu().numpy()
                        y_pred = predicted.cpu().numpy()
                        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                        
                        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{config.num_epochs}] '
                              f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                              f'Loss: {epoch_loss/(batch_idx + 1):.4f} '
                              f'Acc: {100.*correct/total:.2f}% '
                              f'Precision: {100.*precision:.2f}% '
                              f'Recall: {100.*recall:.2f}% '
                              f'F1: {100.*f1:.2f}%')
                
                
                train_loss = epoch_loss / len(train_loader)
                train_acc = 100. * correct / total
                
                
                val_loss, val_acc = evaluate_model(model, test_loader, H, criterion, config)
                
                
                fold_metric['train_loss'].append(train_loss)
                fold_metric['train_acc'].append(train_acc)
                fold_metric['val_loss'].append(val_loss)
                fold_metric['val_acc'].append(val_acc)
            
            
            fold_metrics.append(fold_metric)
            
            
            model.eval()
            accuracy = evaluate_model(model, test_loader, H, criterion, config)[1]
            
            
            if fold == 0 or accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = f'models/best_model_fold_{fold+1}.pt'
                torch.save(model.state_dict(), best_model_path)
                
            fold_data.append((train_loader, test_loader))
        
        
        training_curves_path = os.path.join(plots_dir, 'training_curves.png')
        plot_training_curves(fold_metrics, save_path=training_curves_path)
        print(f"Saved training curves plot to: {training_curves_path}")
        
        
        print(f"\nCreating ensemble model using best model: {best_model_path}")
        ensemble = EnsembleModel(best_model_path).to(config.device)
        ensemble_optimizer = optim.Adam(
            ensemble.parameters(),
            lr=config.learning_rate * 0.1,
            weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        
        ensemble_metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Fine-tune ensemble model on each fold
        for fold, (train_loader, val_loader) in enumerate(fold_data):
            print(f"\nFine-tuning ensemble on fold {fold + 1}")
            ensemble.fine_tune(
                train_loader=train_loader,
                val_loader=val_loader,
                H=H,
                optimizer=ensemble_optimizer,
                criterion=criterion,
                num_epochs=5
            )
            
            
            train_loss, train_acc = evaluate_model(ensemble, train_loader, H, criterion, config)
            val_loss, val_acc = evaluate_model(ensemble, val_loader, H, criterion, config)
            
            ensemble_metrics['train_loss'].append(train_loss)
            ensemble_metrics['train_acc'].append(train_acc)
            ensemble_metrics['val_loss'].append(val_loss)
            ensemble_metrics['val_acc'].append(val_acc)
        
        
        ensemble_perf_path = os.path.join(plots_dir, 'ensemble_performance.png')
        plot_ensemble_performance(ensemble_metrics, save_path=ensemble_perf_path)
        print(f"Saved ensemble performance plot to: {ensemble_perf_path}")
        
        
        ensemble.eval()
        final_loss, final_accuracy = evaluate_model(ensemble, test_loader, H, criterion, config)
        print(f"\nFinal Ensemble Model Accuracy: {final_accuracy:.2f}%")
        
        
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = ensemble(inputs, H)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_scores.extend(probabilities[:, 1].cpu().numpy())  
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        
        print("\nDetailed Evaluation Metrics:")
        print("-" * 30)
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"AUC-ROC: {roc_auc:.4f}")

        
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        roc_df = pd.DataFrame({
            'FPR值': fpr,
            'TPR值': tpr,
            '阈值': thresholds
        })
        roc_df['AUC值'] = ''
        if len(roc_df) > 0:
            roc_df.loc[0, 'AUC值'] = roc_auc
        roc_csv_path = os.path.join(results_dir, 'roc_curve_values.csv')
        roc_df.to_csv(roc_csv_path, index=False, encoding='utf-8-sig')
        print(f"ROC曲线数值已保存到: {roc_csv_path}")

        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        
        roc_plot_path = os.path.join(plots_dir, 'roc_curve.png')
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"Saved ROC curve plot to: {roc_plot_path}")
        
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print("-" * 30)
        print("            Predicted")
        print("             Low  High")
        print(f"Actual Low   {cm[0][0]:3d}   {cm[0][1]:3d}")
        print(f"      High   {cm[1][0]:3d}   {cm[1][1]:3d}")
        
        
        cm_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, ['Low Risk', 'High Risk'], save_path=cm_plot_path)
        print(f"Saved confusion matrix plot to: {cm_plot_path}")
        
        
        torch.save(ensemble.state_dict(), 'models/ensemble_model.pt')
        print(f"\nEnsemble model saved to: models/ensemble_model.pt")
        
        print("\nAll plots have been saved to the 'plots' directory:")
        print(f"1. Hypergraph Statistics: {stats_plot_path}")
        print(f"2. Training Curves: {training_curves_path}")
        print(f"3. Ensemble Performance: {ensemble_perf_path}")
        print(f"4. ROC Curve: {roc_plot_path}")
        print(f"5. Confusion Matrix: {cm_plot_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e

def evaluate_model(model, data_loader, H, criterion, config):
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            outputs = model(inputs, H)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def get_predictions(model, data_loader, H, config):
    
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(config.device)
            outputs = model(inputs, H)
            _, predicted = outputs.max(1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def main():
    train_ensemble()

if __name__ == '__main__':
    main()