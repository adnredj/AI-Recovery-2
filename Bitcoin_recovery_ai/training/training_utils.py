import torch
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from ..models.components.crypto_analyzer import CryptoAnalyzer
from .logging import setup_logger

class TrainingUtils:
    """Utilities for model training"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.crypto_analyzer = CryptoAnalyzer(config)
        
    def prepare_batch(self, 
                     batch: Dict[str, torch.Tensor],
                     device: str = 'cuda') -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare batch for training"""
        inputs = {
            'wallet_data': batch['data'].to(device),
            'version_info': batch['version'].to(device) if 'version' in batch else None
        }
        
        targets = {
            'recovery': batch['recovery'].to(device),
            'pattern': batch['pattern'].to(device),
            'entropy': batch['entropy'].to(device)
        }
        
        return inputs, targets
    
    def calculate_loss(self,
                      predictions: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate training loss"""
        # Recovery loss
        recovery_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions['recovery_probability'],
            targets['recovery']
        )
        
        # Pattern detection loss
        pattern_loss = torch.nn.functional.cross_entropy(
            predictions['encryption_pattern'],
            targets['pattern']
        )
        
        # Entropy prediction loss
        entropy_loss = torch.nn.functional.mse_loss(
            predictions['entropy'],
            targets['entropy']
        )
        
        # Combined loss
        total_loss = (
            self.config.loss_weights.recovery * recovery_loss +
            self.config.loss_weights.pattern_detection * pattern_loss +
            self.config.loss_weights.entropy * entropy_loss
        )
        
        return total_loss, {
            'recovery_loss': recovery_loss.item(),
            'pattern_loss': pattern_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def update_metrics(self,
                      metrics: Dict[str, float],
                      predictions: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update training metrics"""
        with torch.no_grad():
            # Recovery accuracy
            recovery_preds = (predictions['recovery_probability'] > 0.5).float()
            recovery_acc = (recovery_preds == targets['recovery']).float().mean()
            
            # Pattern detection accuracy
            pattern_preds = torch.argmax(predictions['encryption_pattern'], dim=1)
            pattern_acc = (pattern_preds == targets['pattern']).float().mean()
            
            # Entropy error
            entropy_error = torch.abs(predictions['entropy'] - targets['entropy']).mean()
            
            metrics.update({
                'recovery_accuracy': recovery_acc.item(),
                'pattern_accuracy': pattern_acc.item(),
                'entropy_error': entropy_error.item()
            })
            
        return metrics