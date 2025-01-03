import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from pathlib import Path
from ..utils.logging import setup_logger

class WalletPatternDataset(Dataset):
    """Dataset for wallet pattern recognition"""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.logger = setup_logger(__name__)
        
        # Load dataset
        self.samples = self._load_samples()
        self.labels = self._load_labels()
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
        
    def _load_samples(self) -> List[bytes]:
        """Load wallet samples from data directory"""
        samples = []
        for file_path in sorted(self.data_dir.glob('*.dat')):
            try:
                with open(file_path, 'rb') as f:
                    samples.append(f.read())
            except Exception as e:
                self.logger.error(f"Error loading sample {file_path}: {str(e)}")
                
        return samples
        
    def _load_labels(self) -> torch.Tensor:
        """Load corresponding labels"""
        labels = []
        label_file = self.data_dir / 'labels.txt'
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    label_data = line.strip().split(',')
                    labels.append(int(label_data[1]))
        except Exception as e:
            self.logger.error(f"Error loading labels: {str(e)}")
            
        return torch.tensor(labels)