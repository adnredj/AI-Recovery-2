import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional
import os
from pathlib import Path
import json
import numpy as np


__all__ = ['WalletDataset', 'create_train_loader', 'create_test_loader']

class WalletDataset(Dataset):
    """Dataset for wallet recovery training"""
    
    def __init__(self, data_dir: str, config: Dict[str, any], is_training: bool = True):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
            
        self.config = config
        self.is_training = is_training
        
        # Load feature dimensions
        self.feature_dims = {
            'structure': config['features']['structure_dim'],
            'version': config['features']['version_dim'],
            'transaction': config['features']['transaction_dim'],
            'address': config['features']['address_dim'],
            'metadata': config['features']['metadata_dim']
        }
        
        # Load data files
        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(f"No data files found in {self.data_dir}")
        
    def _load_samples(self) -> List[Dict]:
        """Load all training samples"""
        samples = []
        
        # Load from data directory
        for file_path in self.data_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                samples.append(sample)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                
        return samples
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get a training sample"""
        sample = self.samples[idx]
        
        # Process features
        features = {}
        for feature_name, dim in self.feature_dims.items():
            if feature_name in sample['features']:
                feature_data = sample['features'][feature_name]
                feature_tensor = torch.tensor(feature_data, dtype=torch.float32)
                
                # Pad or truncate to expected dimension
                if len(feature_tensor) < dim:
                    padding = torch.zeros(dim - len(feature_tensor))
                    feature_tensor = torch.cat([feature_tensor, padding])
                else:
                    feature_tensor = feature_tensor[:dim]
                    
                features[feature_name] = feature_tensor
            else:
                features[feature_name] = torch.zeros(dim)
                
        # Process labels
        labels = {
            'recovery_type': torch.tensor(sample['labels']['recovery_type'], dtype=torch.long),
            'encryption_type': torch.tensor(sample['labels']['encryption_type'], dtype=torch.long),
            'encryption_prob': torch.tensor(sample['labels']['encryption_prob'], dtype=torch.float32),
            'auth_requirements': torch.tensor(sample['labels']['auth_requirements'], dtype=torch.float32),
            'performance_requirements': torch.tensor(sample['labels']['performance_requirements'], dtype=torch.float32)
        }
        
        return features, labels

def create_train_loader(config: Dict[str, any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation data loaders"""
    
    # Get data directories
    train_dir = Path(config['training']['data']['train_dir'])
    val_dir = Path(config.get('training', {}).get('data', {}).get('val_dir', ''))
    
    # Verify directories exist
    if not train_dir.exists():
        raise ValueError(f"Training directory not found: {train_dir}")
    
    # Create datasets
    train_dataset = WalletDataset(train_dir, config, is_training=True)
    val_dataset = None
    if val_dir.exists():
        val_dataset = WalletDataset(val_dir, config, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['data']['shuffle'],
        num_workers=config['training']['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['data']['num_workers'],
            pin_memory=True
        )
        
    return train_loader, val_loader

def create_test_loader(config: Dict[str, any], features: Dict[str, torch.Tensor]) -> DataLoader:
    """Create a test data loader from a single sample"""
    
    class SingleSampleDataset(Dataset):
        def __init__(self, features):
            self.features = features
            
        def __len__(self):
            return 1
            
        def __getitem__(self, idx):
            return self.features
    
    dataset = SingleSampleDataset(features)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    return loader