import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ProjectDataset(Dataset):
    
    def __init__(self, features, labels, transform=None):
        
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, label

def standardize_individual_responses(data):
    
    standardized_data = np.zeros_like(data, dtype=float)
    
    for i in range(data.shape[0]):
        
        row = data[i].astype(float)
        
        
        mean = np.mean(row)
        std = np.std(row)
        
        
        if std > 0:
            standardized_data[i] = (row - mean) / std
        else:
           
            standardized_data[i] = 0
    
    return standardized_data

def convert_to_binary_labels(labels, method='zero'):
    
    if method == 'median':
        threshold = np.median(labels)
        binary_labels = (labels > threshold).astype(int)
    elif method == 'zero':
        # Use 0 as threshold (above average = 1, below average = 0)
        binary_labels = (labels > 0).astype(int)
    elif method == 'percentile':
        # Use 60th percentile
        threshold = np.percentile(labels, 60)
        binary_labels = (labels > threshold).astype(int)
    
    return binary_labels

def load_data(file_path, feature_cols, label_col):
    
    df = pd.read_csv(file_path)
    
    
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    
    
    X_raw = df[feature_cols].values.astype(float)
    y_raw = df[label_col].values.astype(float)
    
    
    if np.any(np.isnan(X_raw)):
        print("Warning: Features contain missing values, filling with 3.0")
        X_raw = np.nan_to_num(X_raw, nan=3.0)
    
    if np.any(np.isnan(y_raw)):
        print("Warning: Labels contain missing values, filling with 3.0")
        y_raw = np.nan_to_num(y_raw, nan=3.0)
    
    
    print("Performing individual rating standardization...")
    all_data = np.column_stack([X_raw, y_raw])
    all_data_standardized = standardize_individual_responses(all_data)
    
    
    X_standardized = all_data_standardized[:, :-1]
    y_standardized = all_data_standardized[:, -1]
    
    
    print("Converting to binary classification...")
    y_binary = convert_to_binary_labels(y_standardized, method='zero')
    
    
    print("\nBinary classification statistics:")
    print("Method: Values > 0 after standardization → high risk (1), ≤ 0 → low risk (0)")
    low_count = np.sum(y_binary == 0)
    high_count = np.sum(y_binary == 1)
    total = len(y_binary)
    print(f"{label_col}: Low risk={low_count}, High risk={high_count}, "
          f"High risk ratio={high_count/total:.2f}")
    
    return X_standardized, y_binary

def split_data(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    
    train_dataset = ProjectDataset(X_train, y_train)
    test_dataset = ProjectDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader