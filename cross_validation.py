import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from datetime import datetime

from config import Config
from data_loader import load_data
from hypergraph_builder import HypergraphBuilder
from model import HyperGNN
from train import Trainer
from evaluate import Evaluator
from data_augmentation import DataAugmentor

def run_cross_validation(n_splits=3):
    
    config = Config()
    
    
    print("\n[1/5] Loading data...")
    X, y = load_data(config.data_path, config.feature_cols, config.label_col)
    
   
    augmentor = DataAugmentor(random_state=config.random_state)
    
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    
    
    fold_results = []
    
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{n_splits}")
        print('='*50)
        
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        
        print("\nApplying data augmentation...")
        X_train_aug, y_train_aug, _, _ = augmentor.augment_with_validation(
            X_train, y_train, 
            X_val, y_val,
            n_augmented=len(X_train)  
        )
        
        
        print(f"Original training set size: {len(X_train)}")
        print(f"Augmented training set size: {len(X_train_aug)}")
        print(f"Class distribution before augmentation: {np.bincount(y_train)}")
        print(f"Class distribution after augmentation: {np.bincount(y_train_aug)}")
        
        
        X_train_tensor = torch.FloatTensor(X_train_aug)
        y_train_tensor = torch.LongTensor(y_train_aug)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            shuffle=False
        )
        
        
        print("\n[2/5] Building hypergraph...")
        hypergraph_builder = HypergraphBuilder(config.feature_cols, config.factor_groups)
        H, hyperedge_info = hypergraph_builder.build_hypergraph(
            X_train_aug, 
            mi_threshold=config.mi_threshold,
            n_clusters=config.n_clusters
        )
        
        
        print("\n[3/5] Creating model...")
        model = HyperGNN(
            input_dim=len(config.feature_cols),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=config.num_classes
        )
        
        
        print("\n[4/5] Training model...")
        trainer = Trainer(model, config.device, config)
        
        
        fold_save_dir = os.path.join('results', f'fold_{fold}')
        os.makedirs(fold_save_dir, exist_ok=True)
        model_save_path = os.path.join(fold_save_dir, 'best_model.pth')
        
        train_losses, val_losses, val_accuracies = trainer.train(
            train_loader, val_loader, H, config.num_epochs, model_save_path
        )
        
        
        print("\n[5/5] Evaluating model...")
        evaluator = Evaluator(model, config.device, config)
        results = evaluator.evaluate(val_loader, H)
        
        
        fold_results.append({
            'fold': fold,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc'],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        })
        
        
        print(f"\nFold {fold} Results:")
        print("-" * 30)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"AUC-ROC: {results['auc']:.4f}")
    
    
    print("\n" + "="*50)
    print("Cross Validation Results Summary")
    print("="*50)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    avg_results = {}
    std_results = {}
    
    for metric in metrics:
        values = [fold[metric] for fold in fold_results]
        avg_results[metric] = np.mean(values)
        std_results[metric] = np.std(values)
        print(f"\n{metric.capitalize()}:")
        print(f"  Mean: {avg_results[metric]:.4f}")
        print(f"  Std: {std_results[metric]:.4f}")
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join('results', f'cv_results_{timestamp}.json')
    
    full_results = {
        'config': config.__dict__,
        'fold_results': fold_results,
        'average_results': avg_results,
        'std_results': std_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")
    
    return avg_results, std_results, fold_results

if __name__ == "__main__":
    run_cross_validation(n_splits=3) 