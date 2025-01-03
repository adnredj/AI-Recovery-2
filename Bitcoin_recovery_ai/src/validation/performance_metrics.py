from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
from datetime import datetime
from ..utils.logging import setup_logger

class RecoveryMetrics:
    """Metrics calculator for wallet recovery system"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.metrics_history = []
        
    def _calculate_recovery_metrics(self,
                                 recovery_results: List[Dict],
                                 true_keys: List[str]) -> Dict[str, float]:
        """Calculate key recovery specific metrics
        
        Args:
            recovery_results: List of recovery attempt results
            true_keys: List of true key values
            
        Returns:
            Dictionary containing recovery metrics
        """
        try:
            metrics = {
                'recovery_success_rate': 0.0,
                'average_attempts': 0.0,
                'time_to_recovery': 0.0,
                'accuracy': 0.0,
                'partial_matches': 0.0,
                'false_positives': 0.0,
                'confidence_correlation': 0.0
            }
            
            if not recovery_results or not true_keys:
                return metrics
                
            # Calculate success rate
            successful_recoveries = sum(
                1 for result in recovery_results
                if result.get('recovered_key') in true_keys
            )
            metrics['recovery_success_rate'] = successful_recoveries / len(recovery_results)
            
            # Calculate average attempts
            total_attempts = sum(
                result.get('attempts', 1)
                for result in recovery_results
            )
            metrics['average_attempts'] = total_attempts / len(recovery_results)
            
            # Calculate time metrics
            recovery_times = [
                result.get('recovery_time', 0)
                for result in recovery_results
                if result.get('recovered_key') in true_keys
            ]
            if recovery_times:
                metrics['time_to_recovery'] = np.mean(recovery_times)
                
            # Calculate accuracy
            predicted_keys = [
                result.get('recovered_key', '')
                for result in recovery_results
            ]
            metrics['accuracy'] = self._calculate_key_accuracy(
                predicted_keys,
                true_keys
            )
            
            # Calculate partial matches
            metrics['partial_matches'] = self._calculate_partial_matches(
                predicted_keys,
                true_keys
            )
            
            # Calculate false positives
            metrics['false_positives'] = self._calculate_false_positives(
                predicted_keys,
                true_keys
            )
            
            # Calculate confidence correlation
            metrics['confidence_correlation'] = self._calculate_confidence_correlation(
                recovery_results,
                true_keys
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating recovery metrics: {str(e)}")
            return metrics
            
    def _calculate_pattern_metrics(self,
                                pattern_predictions: np.ndarray,
                                true_patterns: np.ndarray) -> Dict[str, float]:
        """Calculate pattern detection metrics
        
        Args:
            pattern_predictions: Predicted pattern array
            true_patterns: True pattern array
            
        Returns:
            Dictionary containing pattern metrics
        """
        try:
            metrics = {
                'pattern_accuracy': 0.0,
                'pattern_precision': 0.0,
                'pattern_recall': 0.0,
                'pattern_f1': 0.0,
                'pattern_support': 0.0,
                'pattern_confidence': 0.0
            }
            
            if pattern_predictions.size == 0 or true_patterns.size == 0:
                return metrics
                
            # Calculate basic metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                true_patterns,
                pattern_predictions,
                average='weighted'
            )
            
            metrics['pattern_precision'] = precision
            metrics['pattern_recall'] = recall
            metrics['pattern_f1'] = f1
            metrics['pattern_support'] = np.mean(support)
            
            # Calculate accuracy
            metrics['pattern_accuracy'] = accuracy_score(
                true_patterns,
                pattern_predictions
            )
            
            # Calculate confidence
            if hasattr(pattern_predictions, 'probability'):
                metrics['pattern_confidence'] = np.mean(
                    pattern_predictions.probability
                )
            else:
                metrics['pattern_confidence'] = np.mean(
                    pattern_predictions == true_patterns
                )
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern metrics: {str(e)}")
            return metrics
            
    def _calculate_version_metrics(self,
                                version_scores: Dict[str, float],
                                true_version: str) -> Dict[str, float]:
        """Calculate version detection metrics
        
        Args:
            version_scores: Dictionary of version predictions and scores
            true_version: True version string
            
        Returns:
            Dictionary containing version metrics
        """
        try:
            metrics = {
                'version_accuracy': 0.0,
                'version_confidence': 0.0,
                'version_rank': 0.0,
                'version_distance': 0.0
            }
            
            if not version_scores or not true_version:
                return metrics
                
            # Sort versions by confidence score
            sorted_versions = sorted(
                version_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Calculate accuracy (exact match)
            predicted_version = sorted_versions[0][0]
            metrics['version_accuracy'] = float(predicted_version == true_version)
            
            # Calculate confidence
            metrics['version_confidence'] = version_scores.get(true_version, 0.0)
            
            # Calculate rank of true version
            for rank, (version, _) in enumerate(sorted_versions):
                if version == true_version:
                    metrics['version_rank'] = rank + 1
                    break
                    
            # Calculate version distance
            metrics['version_distance'] = self._calculate_version_distance(
                predicted_version,
                true_version
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating version metrics: {str(e)}")
            return metrics
            
    def _calculate_key_accuracy(self,
                             predicted_keys: List[str],
                             true_keys: List[str]) -> float:
        """Calculate accuracy of key recovery"""
        correct = sum(1 for pred in predicted_keys if pred in true_keys)
        return correct / len(predicted_keys) if predicted_keys else 0.0
        
    def _calculate_partial_matches(self,
                                predicted_keys: List[str],
                                true_keys: List[str],
                                threshold: float = 0.8) -> float:
        """Calculate partial key matches"""
        partial_matches = 0
        for pred in predicted_keys:
            for true in true_keys:
                similarity = self._calculate_key_similarity(pred, true)
                if similarity >= threshold:
                    partial_matches += 1
                    break
        return partial_matches / len(predicted_keys) if predicted_keys else 0.0
        
    def _calculate_false_positives(self,
                                predicted_keys: List[str],
                                true_keys: List[str]) -> float:
        """Calculate false positive rate"""
        false_positives = sum(1 for pred in predicted_keys if pred not in true_keys)
        return false_positives / len(predicted_keys) if predicted_keys else 0.0
        
    def _calculate_confidence_correlation(self,
                                      recovery_results: List[Dict],
                                      true_keys: List[str]) -> float:
        """Calculate correlation between confidence and correctness"""
        confidences = []
        correctness = []
        
        for result in recovery_results:
            if 'confidence' in result and 'recovered_key' in result:
                confidences.append(result['confidence'])
                correctness.append(float(result['recovered_key'] in true_keys))
                
        if confidences and correctness:
            return np.corrcoef(confidences, correctness)[0, 1]
        return 0.0
        
    def _calculate_version_distance(self,
                                 version1: str,
                                 version2: str) -> float:
        """Calculate semantic distance between versions"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad versions to same length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            # Calculate weighted distance
            weights = [0.5 ** i for i in range(max_len)]
            distance = sum(
                w * abs(a - b)
                for w, a, b in zip(weights, v1_parts, v2_parts)
            )
            
            return distance
            
        except ValueError:
            return 1.0  # Maximum distance for invalid versions