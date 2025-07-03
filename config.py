
import torch

class Config:
    
    data_path = 'your_data_path'
    test_size = 0.2  
    random_state = 42
    
    #for example
    feature_cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
                    'C1', 'C2', 'C3',
                    'D1', 'D2', 'D3',
                    'E1', 'E2', 'E3',
                    'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
                    'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8']
    
    label_col = 'Delay'  
    
    
    factor_groups = {
        'A': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        'B': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
        'C': ['C1', 'C2', 'C3'],
        'D': ['D1', 'D2', 'D3'],
        'E': ['E1', 'E2', 'E3'],
        'F': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
        'G': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8']
    }
    
    
    mi_threshold = 0.32         
    n_clusters = 10             
    
    
    hidden_dim = 128
    num_layers = 4
    dropout = 0.5             
    num_classes = 2           
    
    
    batch_size = 16           
    learning_rate = 0.0001    
    momentum = 0.9         
    weight_decay = 5e-3    
    num_epochs = 200
    early_stopping_patience = 10
    num_folds = 3
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    model_save_path = 'best_model.pth'
    results_save_path = 'results/'
