from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from ..utils.logging import setup_logger

class RecoveryMetrics:
    """Performance metrics for bitcoin wallet recovery"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.history = []
        self.current_epoch = 0
        
    def calculate_metrics(self,
                         predictions: torch.Tensor,
                         targets: torch.Tensor,
                         validation_data: Dict[str, any]) -> Dict[str, float]:
        """Calculate comprehensive recovery metrics"""
        metrics = {}
        
        # Basic accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(predictions, targets)
        metrics.update(accuracy_metrics)
        
        # Recovery specific metrics
        recovery_metrics = self._calculate_recovery_metrics(predictions, validation_data)
        metrics.update(recovery_metrics)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(predictions)
        metrics.update(performance_metrics)
        
        # Update history
        self._update_history(metrics)
        
        return metrics
    
    def _calculate_accuracy_metrics(self,
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor) -> Dict[str, float]:
        """Calculate basic accuracy metrics"""
        # Convert to numpy for sklearn metrics
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_np,
            pred_np.argmax(axis=1),
            average='weighted'
        )
        
        # Calculate AUC-ROC if applicable
        try:
            auc_roc = roc_auc_score(target_np, pred_np, multi_class='ovr')
        except:
            auc_roc = None
            
        return {
            'accuracy': (pred_np.argmax(axis=1) == target_np).mean(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
    
    def _calculate_recovery_metrics(self,
                                  predictions: torch.Tensor,
                                  validation_data: Dict[str, any]) -> Dict[str, float]:
        """Calculate recovery-specific metrics"""
        metrics = {}
        
        # Key recovery rate
        key_recovery = self._calculate_key_recovery_rate(
            predictions,
            validation_data.get('keys', None)
        )
        metrics['key_recovery_rate'] = key_recovery
        
        # Address recovery accuracy
        address_accuracy = self._calculate_address_accuracy(
            predictions,
            validation_data.get('addresses', None)
        )
        metrics['address_accuracy'] = address_accuracy
        
        # Transaction validation rate
        transaction_rate = self._calculate_transaction_validation_rate(
            predictions,
            validation_data.get('transactions', None)
        )
        metrics['transaction_validation_rate'] = transaction_rate
        
        # Pattern match score
        pattern_score = self._calculate_pattern_match_score(
            predictions,
            validation_data.get('patterns', None)
        )
        metrics['pattern_match_score'] = pattern_score
        
        return metrics
    
    def _calculate_performance_metrics(self,
                                    predictions: torch.Tensor) -> Dict[str, float]:
        """Calculate model performance metrics"""
        metrics = {}
        
        # Confidence scores
        confidence_metrics = self._calculate_confidence_metrics(predictions)
        metrics.update(confidence_metrics)
        
        # Entropy metrics
        entropy_metrics = self._calculate_entropy_metrics(predictions)
        metrics.update(entropy_metrics)
        
        # Stability metrics
        stability_metrics = self._calculate_stability_metrics(predictions)
        metrics.update(stability_metrics)
        
        return metrics
    
    def _calculate_key_recovery_rate(self,
                                   predictions: torch.Tensor,
                                   target_keys: Optional[torch.Tensor]) -> float:
        """Calculate key recovery success rate"""
        if target_keys is None:
            return 0.0
            
        # Convert predictions to key format
        predicted_keys = self._convert_to_keys(predictions)
        
        # Calculate match rate
        match_rate = (predicted_keys == target_keys).float().mean().item()
        
        return match_rate
    
    def _calculate_address_accuracy(self,
                                  predictions: torch.Tensor,
                                  target_addresses: Optional[torch.Tensor]) -> float:
        """Calculate address generation accuracy"""
        if target_addresses is None:
            return 0.0
            
        # Generate addresses from predictions
        predicted_addresses = self._generate_addresses(predictions)
        
        # Calculate accuracy
        accuracy = (predicted_addresses == target_addresses).float().mean().item()
        
        return accuracy
    
    def _calculate_confidence_metrics(self,
                                   predictions: torch.Tensor) -> Dict[str, float]:
        """Calculate prediction confidence metrics"""
        # Get probability distributions
        probs = torch.softmax(predictions, dim=-1)
        
        # Calculate max probabilities
        max_probs = probs.max(dim=-1)[0]
        
        return {
            'mean_confidence': max_probs.mean().item(),
            'min_confidence': max_probs.min().item(),
            'max_confidence': max_probs.max().item()
        }
    
    def _calculate_entropy_metrics(self,
                                 predictions: torch.Tensor) -> Dict[str, float]:
        """Calculate entropy-based metrics"""
        # Get probability distributions
        probs = torch.softmax(predictions, dim=-1)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        return {
            'mean_entropy': entropy.mean().item(),
            'entropy_std': entropy.std().item()
        }
    
    def _calculate_stability_metrics(self,
                                   predictions: torch.Tensor) -> Dict[str, float]:
        """Calculate prediction stability metrics"""
        # Calculate variance across batch
        pred_var = torch.var(predictions, dim=0)
        
        return {
            'prediction_variance': pred_var.mean().item(),
            'stability_score': 1.0 - min(pred_var.mean().item(), 1.0)
        }
    
    def _update_history(self, metrics: Dict[str, float]):
        """Update metrics history"""
        self.history.append({
            'epoch': self.current_epoch,
            'timestamp': datetime.now(),
            'metrics': metrics.copy()
        })
        self.current_epoch += 1
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of metrics history"""
        if not self.history:
            return {}
            
        summary = {
            'epochs': len(self.history),
            'latest': self.history[-1]['metrics'],
            'best': self._get_best_metrics(),
            'trends': self._calculate_trends()
        }
        
        return summary
    
    def _get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics across all epochs"""
        best_metrics = {}
        
        for metric in self.history[0]['metrics'].keys():
            values = [h['metrics'][metric] for h in self.history]
            best_metrics[metric] = max(values)
            
        return best_metrics
    
    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate metric trends"""
        if len(self.history) < 2:
            return {}
            
        trends = {}
        recent_window = 5
        
        for metric in self.history[0]['metrics'].keys():
            values = [h['metrics'][metric] for h in self.history[-recent_window:]]
            trend = self._determine_trend(values)
            trends[metric] = trend
            
        return trends
    
    def _determine_trend(self, values: List[float]) -> str:
        """Determine trend direction from values"""
        if len(values) < 2:
            return 'stable'
            
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if abs(slope) < self.config.trend_threshold:
            return 'stable'
        return 'increasing' if slope > 0 else 'decreasing'