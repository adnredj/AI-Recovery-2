import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch

def calculate_metrics(targets: Union[np.ndarray, List], 
                     predictions: Union[np.ndarray, List],
                     threshold: float = 0.5) -> Dict[str, float]:
    """Calculate various classification metrics
    
    Args:
        targets: Ground truth labels
        predictions: Model predictions
        threshold: Classification threshold for binary predictions
        
    Returns:
        Dictionary containing calculated metrics
    """
    # Convert inputs to numpy arrays if needed
    if isinstance(targets, (list, torch.Tensor)):
        targets = np.array(targets)
    if isinstance(predictions, (list, torch.Tensor)):
        predictions = np.array(predictions)
        
    # Apply threshold for binary predictions if needed
    if predictions.dtype == np.float32 or predictions.dtype == np.float64:
        predictions = (predictions > threshold).astype(np.int32)
        
    # Calculate basic metrics
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, 
        predictions, 
        average='binary'
    )
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'npv': float(npv),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

def calculate_recovery_metrics(predictions: torch.Tensor,
                             targets: torch.Tensor,
                             threshold: float = 0.5) -> Dict[str, float]:
    """Calculate metrics specific to wallet recovery
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        threshold: Classification threshold
        
    Returns:
        Dictionary containing recovery-specific metrics
    """
    # Convert to numpy and ensure binary values
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    # Get basic classification metrics
    base_metrics = calculate_metrics(targets, predictions, threshold)
    
    # Calculate recovery-specific metrics
    total_wallets = len(targets)
    recovered_wallets = base_metrics['true_positives']
    false_recoveries = base_metrics['false_positives']
    
    recovery_rate = recovered_wallets / total_wallets if total_wallets > 0 else 0.0
    false_recovery_rate = false_recoveries / total_wallets if total_wallets > 0 else 0.0
    
    # Combine metrics
    recovery_metrics = {
        **base_metrics,
        'recovery_rate': float(recovery_rate),
        'false_recovery_rate': float(false_recovery_rate),
        'total_wallets': int(total_wallets),
        'recovered_wallets': int(recovered_wallets),
        'false_recoveries': int(false_recoveries)
    }
    
    return recovery_metrics

def calculate_pattern_metrics(predictions: torch.Tensor,
                            targets: torch.Tensor,
                            num_classes: Optional[int] = None) -> Dict[str, float]:
    """Calculate metrics for pattern classification
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth targets
        num_classes: Number of pattern classes
        
    Returns:
        Dictionary containing pattern-specific metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    # Convert logits to class predictions if needed
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
        
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        labels=range(num_classes) if num_classes else None
    )
    
    # Calculate overall accuracy
    accuracy = accuracy_score(targets, predictions)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(np.mean(f1)),
        'weighted_f1': float(np.average(f1, weights=support)),
        'per_class_metrics': {
            f'class_{i}': {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f),
                'support': int(s)
            }
            for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support))
        }
    }
    
    return metrics

def calculate_entropy_metrics(predictions: torch.Tensor,
                            targets: torch.Tensor) -> Dict[str, float]:
    """Calculate metrics for entropy prediction
    
    Args:
        predictions: Model entropy predictions
        targets: Ground truth entropy values
        
    Returns:
        Dictionary containing entropy-specific metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    # Calculate regression metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    # Calculate R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation),
        'r2': float(r2)
    }