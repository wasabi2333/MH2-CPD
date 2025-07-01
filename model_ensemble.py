import torch
import torch.nn as nn
import numpy as np
from model import HyperGNN
from config import Config

class EnsembleModel(nn.Module):
    def __init__(self, base_model_path, num_models=3):
        
        super().__init__()
        self.config = Config()
        self.num_models = num_models
        self.models = nn.ModuleList()
        
        
        base_model = HyperGNN(
            input_dim=len(self.config.feature_cols),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_classes=self.config.num_classes
        )
        
        
        state_dict = torch.load(base_model_path)
        
        
        if any(k.startswith('models.') for k in state_dict.keys()):
            
            base_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('models.0.'):
                    base_state_dict[k[8:]] = v  
        else:
            
            base_state_dict = {}
            for k, v in state_dict.items():
                
                clean_key = k[1:] if k.startswith('.') else k
                base_state_dict[clean_key] = v
        
        
        base_model.load_state_dict(base_state_dict)
        self.models.append(base_model)
        
        
        for i in range(num_models - 1):
            model = HyperGNN(
                input_dim=len(self.config.feature_cols),
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_classes=self.config.num_classes
            )
            
            model.load_state_dict(base_model.state_dict())
            with torch.no_grad():
                for param in model.parameters():
                    
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
            self.models.append(model)
    
    def forward(self, x, H):
        
        predictions = []
        for model in self.models:
            pred = model(x, H)
            predictions.append(pred)
        
        stacked_preds = torch.stack(predictions, dim=0)
        
        
        softmax_preds = torch.softmax(stacked_preds, dim=-1)  
        entropy = -(softmax_preds * torch.log(softmax_preds + 1e-10)).sum(dim=-1)  
        weights = torch.softmax(-entropy, dim=0)  
        
        weights = weights.unsqueeze(-1)  
        ensemble_pred = (stacked_preds * weights).sum(dim=0)  
        
        return ensemble_pred
    
    def fine_tune(self, train_loader, val_loader, H, optimizer, criterion, num_epochs=10):
        
        for epoch in range(num_epochs):
            
            self.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                optimizer.zero_grad()
                output = self(batch_x, H)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            
            self.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.config.device)
                    batch_y = batch_y.to(self.config.device)
                    
                    output = self(batch_x, H)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = output.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}')
            print(f'Val Accuracy: {100.*correct/total:.2f}%\n') 