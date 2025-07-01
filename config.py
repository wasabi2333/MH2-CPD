
import torch

class Config:
    
    data_path = 'data_229_latest.csv'
    test_size = 0.2  
    random_state = 42
    
    
    feature_cols = ['CIS1', 'CIS2', 'CIS3', 'CIS4', 'CIS5', 'CIS6',
                    'RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'RA6', 'RA7', 'RA8',
                    'ITM1', 'ITM2', 'ITM3',
                    'CRAA1', 'CRAA2', 'CRAA3',
                    'CS1', 'CS2', 'CS3',
                    'PE1', 'PE2', 'PE3', 'PE4', 'PE5', 'PE6',
                    'EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE6', 'EE7', 'EE8']
    
    label_col = 'PP1'  
    
    
    factor_groups = {
        'CIS': ['CIS1', 'CIS2', 'CIS3', 'CIS4', 'CIS5', 'CIS6'],
        'RA': ['RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'RA6', 'RA7', 'RA8'],
        'ITM': ['ITM1', 'ITM2', 'ITM3'],
        'CRAA': ['CRAA1', 'CRAA2', 'CRAA3'],
        'CS': ['CS1', 'CS2', 'CS3'],
        'PE': ['PE1', 'PE2', 'PE3', 'PE4', 'PE5', 'PE6'],
        'EE': ['EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE6', 'EE7', 'EE8']
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