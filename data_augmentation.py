import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class DataAugmentor:
    def __init__(self, random_state=42):
        
        self.random_state = random_state
        
    def add_gaussian_noise(self, X, noise_factor=0.05):
        
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise
    
    def random_scale(self, X, scale_range=(0.95, 1.05)):
        
        scales = np.random.uniform(scale_range[0], scale_range[1], size=(X.shape[1],))
        return X * scales
    
    def smote_oversampling(self, X, y):
        
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def augment(self, X, y, n_augmented=None):
        
        if n_augmented is None:
            n_augmented = len(X)
        
        
        X_smote, y_smote = self.smote_oversampling(X, y)
        
        
        n_samples = len(X_smote)
        indices = np.random.choice(n_samples, size=n_augmented, replace=True)
        
        X_aug = X_smote[indices].copy()
        y_aug = y_smote[indices].copy()
        
        
        X_aug = self.add_gaussian_noise(X_aug)
        
        
        X_aug = self.random_scale(X_aug)
        
        
        X_combined = np.vstack([X, X_aug])
        y_combined = np.concatenate([y, y_aug])
        
        
        shuffle_idx = np.random.permutation(len(X_combined))
        X_combined = X_combined[shuffle_idx]
        y_combined = y_combined[shuffle_idx]
        
        return X_combined, y_combined
    
    def augment_with_validation(self, X_train, y_train, X_val=None, y_val=None, n_augmented=None):
        
        
        X_train_aug, y_train_aug = self.augment(X_train, y_train, n_augmented)
        
        if X_val is not None and y_val is not None:
            return X_train_aug, y_train_aug, X_val, y_val
        else:
            return X_train_aug, y_train_aug 