import torch
from typing import Dict, List, Optional
from ..utils.logging import setup_logger
from ..utils.crypto_utils import verify_key_recovery

class WalletRecoveryValidator:
    """Validator for wallet recovery model"""
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.logger = setup_logger(__name__)
        
    def validate(self, model, dataloader) -> Dict[str, float]:
        """Validate model performance"""
        model.eval()
        validation_metrics = {
            'recovery_success_rate': 0.0,
            'pattern_detection_accuracy': 0.0,
            'key_derivation_accuracy': 0.0,
            'false_positive_rate': 0.0
        }
        
        with torch.no_grad():
            for batch in dataloader:
                wallet_data = batch['data'].to(self.device)
                version_info = batch['version'].to(self.device)
                ground_truth = batch['ground_truth']
                
                # Get model predictions
                predictions = model(wallet_data, version_info)
                
                # Validate recovery success
                recovery_success = self._validate_recovery(
                    predictions['recovery_results'],
                    ground_truth['keys']
                )
                
                # Validate pattern detection
                pattern_accuracy = self._validate_patterns(
                    predictions['encryption_pattern'],
                    ground_truth['patterns']
                )
                
                # Update metrics
                validation_metrics['recovery_success_rate'] += recovery_success
                validation_metrics['pattern_detection_accuracy'] += pattern_accuracy
                
        # Average metrics
        for key in validation_metrics:
            validation_metrics[key] /= len(dataloader)
            
        return validation_metrics
    
    def _validate_recovery(self, 
                         recovery_results: List[Dict],
                         ground_truth_keys: List[str]) -> float:
        """Validate key recovery results"""
        successful_recoveries = 0
        total_attempts = len(ground_truth_keys)
        
        for pred, truth in zip(recovery_results, ground_truth_keys):
            if verify_key_recovery(pred['recovered_key'], truth):
                successful_recoveries += 1
                
        return successful_recoveries / total_attempts
    
    def _validate_patterns(self,
                         predicted_patterns: torch.Tensor,
                         true_patterns: torch.Tensor) -> float:
        """Validate pattern detection accuracy"""
        pred_classes = torch.argmax(predicted_patterns, dim=1)
        return (pred_classes == true_patterns).float().mean().item()