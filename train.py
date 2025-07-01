import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os

class Trainer:
   
    
    def __init__(self, model, device, config, class_weights=None):
        
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        
        if class_weights is not None:
            weights = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True
        )
        self.scheduler = None
        
        
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []
        
        
        self.scheduler = None
        
    def train_epoch(self, train_loader, H):
        
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        H = H.to(self.device)
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (features, labels) in enumerate(progress_bar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            
            outputs = self.model(features, H)
            
            
            loss = self.criterion(outputs, labels)
            
            
            self.optimizer.zero_grad()
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        return total_loss / n_batches
    
    def evaluate(self, test_loader, H):
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        all_predictions = []
        all_labels = []
        
        H = H.to(self.device)
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                
                outputs = self.model(features, H)
                
               
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                n_batches += 1
                
               
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        avg_loss = total_loss / n_batches
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, H, num_epochs, save_path):
        
        steps_per_epoch = len(train_loader)
        half_epoch = num_epochs // 2
        self.scheduler_adam = OneCycleLR(
            self.optimizer_adam,
            max_lr=self.config.learning_rate,
            epochs=half_epoch,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.25,
            div_factor=8.0,
            final_div_factor=1e2,
            anneal_strategy='cos',
            three_phase=True
        )
        self.scheduler_sgd = OneCycleLR(
            self.optimizer_sgd,
            max_lr=self.config.learning_rate * 0.1,
            epochs=num_epochs - half_epoch,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.25,
            div_factor=8.0,
            final_div_factor=1e2,
            anneal_strategy='cos',
            three_phase=True
        )
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            if epoch < half_epoch:
                self.optimizer = self.optimizer_adam
                self.scheduler = self.scheduler_adam
            else:
                self.optimizer = self.optimizer_sgd
                self.scheduler = self.scheduler_sgd
            print(f'\n{"="*20} Epoch {epoch+1}/{num_epochs} {"="*20}')
            
            
            train_loss = self.train_epoch(train_loader, H)
            self.train_losses.append(train_loss)
            
            
            val_loss, val_accuracy = self.evaluate(val_loader, H)
            self.test_losses.append(val_loss)
            self.test_accuracies.append(val_accuracy)
            
            
            print(f'\nEpoch Summary:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }
                print(f'\n>>> New best model saved! (validation loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                print(f'\n>>> No improvement. Patience: {patience_counter}/{self.config.early_stopping_patience}')
                print(f'>>> Current best: {best_val_loss:.4f} (epoch {best_epoch+1})')
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f'\n!!! Early stopping triggered at epoch {epoch+1} !!!')
                print(f'!!! Best model was at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f} !!!')
                break
        
        
        if best_model_state is not None:
            torch.save(best_model_state, save_path)
            
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f'\nFinal model restored to best checkpoint from epoch {best_epoch+1}')
        
        return self.train_losses, self.test_losses, self.test_accuracies