import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from ..utils.logging import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelEvaluator:
    """Comprehensive model evaluation and testing"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = setup_logger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_model(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, any]:
        """Perform comprehensive model evaluation"""
        self.model.eval()
        
        # Collect predictions and ground truth
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get model predictions
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        # Calculate metrics
        evaluation_results = {
            'classification_metrics': self._calculate_classification_metrics(
                all_targets, all_predictions
            ),
            'confusion_matrix': self._generate_confusion_matrix(
                all_targets, all_predictions
            ),
            'roc_curves': self._generate_roc_curves(
                all_targets, all_probabilities
            ),
            'error_analysis': self._perform_error_analysis(
                all_targets, all_predictions, all_probabilities
            ),
            'confidence_analysis': self._analyze_prediction_confidence(
                all_probabilities
            )
        }
        
        # Generate visualizations
        self._generate_evaluation_plots(evaluation_results)
        
        return evaluation_results
    
    def test_specific_patterns(self, test_data: List[bytes]) -> Dict[str, any]:
        """Test model on specific wallet patterns"""
        self.model.eval()
        results = []
        
        for data in test_data:
            # Preprocess data
            processed_data = self._preprocess_test_data(data)
            
            with torch.no_grad():
                # Get model predictions
                outputs = self.model(processed_data)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Analyze predictions
                pattern_analysis = self._analyze_pattern_prediction(
                    data, outputs, probabilities
                )
                results.append(pattern_analysis)
                
        return {
            'individual_results': results,
            'summary': self._summarize_pattern_tests(results)
        }
    
    def _calculate_classification_metrics(self, 
                                       targets: List[int], 
                                       predictions: List[int]) -> Dict[str, any]:
        """Calculate detailed classification metrics"""
        # Get classification report
        report = classification_report(
            targets, predictions, 
            target_names=self.config.pattern_classes,
            output_dict=True
        )
        
        # Calculate additional metrics
        metrics = {
            'classification_report': report,
            'per_class_accuracy': self._calculate_per_class_accuracy(
                targets, predictions
            ),
            'balanced_accuracy': self._calculate_balanced_accuracy(
                targets, predictions
            ),
            'error_rate': 1 - np.mean(np.array(targets) == np.array(predictions))
        }
        
        return metrics
    
    def _generate_confusion_matrix(self, 
                                 targets: List[int], 
                                 predictions: List[int]) -> Dict[str, any]:
        """Generate and analyze confusion matrix"""
        # Calculate confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return {
            'raw_matrix': cm,
            'normalized_matrix': cm_normalized,
            'class_names': self.config.pattern_classes,
            'misclassification_analysis': self._analyze_misclassifications(cm)
        }
    
    def _generate_roc_curves(self, 
                           targets: List[int], 
                           probabilities: List[List[float]]) -> Dict[str, any]:
        """Generate ROC curves and related metrics"""
        from sklearn.metrics import roc_curve, auc
        
        roc_data = {}
        probabilities = np.array(probabilities)
        
        # Calculate ROC curve for each class
        for i, class_name in enumerate(self.config.pattern_classes):
            # Convert to binary classification problem
            binary_targets = (np.array(targets) == i).astype(int)
            class_probs = probabilities[:, i]
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(binary_targets, class_probs)
            roc_auc = auc(fpr, tpr)
            
            roc_data[class_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
            
        return roc_data
    
    def _perform_error_analysis(self, 
                              targets: List[int], 
                              predictions: List[int],
                              probabilities: List[List[float]]) -> Dict[str, any]:
        """Detailed error analysis"""
        errors = {
            'misclassified_indices': np.where(np.array(targets) != np.array(predictions))[0],
            'error_distribution': self._analyze_error_distribution(targets, predictions),
            'confidence_analysis': self._analyze_error_confidence(
                targets, predictions, probabilities
            ),
            'common_patterns': self._identify_error_patterns(targets, predictions)
        }
        
        return errors
    
    def _analyze_prediction_confidence(self, 
                                    probabilities: List[List[float]]) -> Dict[str, any]:
        """Analyze prediction confidence patterns"""
        probabilities = np.array(probabilities)
        
        return {
            'mean_confidence': float(np.mean(np.max(probabilities, axis=1))),
            'confidence_distribution': self._analyze_confidence_distribution(probabilities),
            'uncertainty_analysis': self._analyze_prediction_uncertainty(probabilities),
            'calibration_metrics': self._calculate_calibration_metrics(probabilities)
        }
    
    def _generate_evaluation_plots(self, results: Dict[str, any]):
        """Generate evaluation visualization plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_dir = Path(self.config.output_dir) / 'evaluation_plots' / timestamp
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix plot
        self._plot_confusion_matrix(
            results['confusion_matrix']['normalized_matrix'],
            results['confusion_matrix']['class_names'],
            plot_dir / 'confusion_matrix.png'
        )
        
        # ROC curves plot
        self._plot_roc_curves(
            results['roc_curves'],
            plot_dir / 'roc_curves.png'
        )
        
        # Confidence distribution plot
        self._plot_confidence_distribution(
            results['confidence_analysis']['confidence_distribution'],
            plot_dir / 'confidence_distribution.png'
        )
        
        # Error analysis plots
        self._plot_error_analysis(
            results['error_analysis'],
            plot_dir / 'error_analysis.png'
        )
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: Path):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f', 
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _preprocess_test_data(self, data: bytes) -> torch.Tensor:
        """Preprocess test data for model input
        
        Args:
            data: Raw wallet data bytes
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Convert bytes to numpy array
            byte_array = np.frombuffer(data, dtype=np.uint8)
            
            # Normalize data
            normalized = byte_array.astype(np.float32) / 255.0
            
            # Extract features
            features = []
            
            # Statistical features
            stats = [
                np.mean(normalized),
                np.std(normalized),
                np.median(normalized),
                np.percentile(normalized, [25, 75])
            ]
            features.extend(stats)
            
            # Entropy features
            hist, _ = np.histogram(normalized, bins=256, density=True)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features.append(entropy)
            
            # Pattern features
            for window_size in [2, 4, 8]:
                pattern_counts = self._count_patterns(normalized, window_size)
                features.append(pattern_counts)
                
            # Convert to tensor
            tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Reshape for model input
            tensor = tensor.view(1, -1, self.config.input_length)
            
            return tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing test data: {str(e)}")
            return None
    
    def _analyze_pattern_prediction(self, 
                                  data: bytes, 
                                  outputs: torch.Tensor,
                                  probabilities: torch.Tensor) -> Dict[str, any]:
        """Analyze individual pattern predictions
        
        Args:
            data: Original wallet data
            outputs: Model output predictions
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary containing detailed prediction analysis
        """
        try:
            analysis = {
                'predictions': {},
                'confidence': {},
                'patterns': {},
                'anomalies': []
            }
            
            # Get top predictions
            top_k = min(3, len(self.config.pattern_classes))
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Analyze top predictions
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                
                analysis['predictions'][self.config.pattern_classes[class_idx]] = {
                    'probability': prob,
                    'rank': i + 1,
                    'confidence_margin': prob - top_probs[0][i+1].item() if i < top_k-1 else prob
                }
                
            # Confidence analysis
            analysis['confidence'] = {
                'overall': float(torch.max(probabilities).item()),
                'entropy': float(-torch.sum(probabilities * torch.log2(probabilities + 1e-10))),
                'margin': float(top_probs[0][0] - top_probs[0][1])
            }
            
            # Pattern analysis
            analysis['patterns'] = self._analyze_detected_patterns(data, outputs)
            
            # Anomaly detection
            analysis['anomalies'] = self._detect_prediction_anomalies(
                outputs, probabilities, data
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction: {str(e)}")
            return {}
    
    def _summarize_pattern_tests(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Summarize pattern test results
        
        Args:
            results: List of individual test results
            
        Returns:
            Dictionary containing test summary
        """
        try:
            summary = {
                'overall': {},
                'per_pattern': {},
                'confidence': {},
                'anomalies': {}
            }
            
            # Overall statistics
            total_tests = len(results)
            high_confidence = sum(1 for r in results 
                                if r['confidence']['overall'] > self.config.confidence_threshold)
            
            summary['overall'] = {
                'total_tests': total_tests,
                'high_confidence_ratio': high_confidence / total_tests,
                'mean_confidence': np.mean([r['confidence']['overall'] for r in results]),
                'success_rate': self._calculate_success_rate(results)
            }
            
            # Per-pattern statistics
            for pattern in self.config.pattern_classes:
                pattern_results = [r for r in results 
                                 if pattern in r['predictions'] and 
                                 r['predictions'][pattern]['rank'] == 1]
                
                if pattern_results:
                    summary['per_pattern'][pattern] = {
                        'count': len(pattern_results),
                        'mean_confidence': np.mean([r['confidence']['overall'] 
                                                  for r in pattern_results]),
                        'success_rate': len(pattern_results) / total_tests
                    }
                    
            # Confidence analysis
            all_confidences = [r['confidence']['overall'] for r in results]
            summary['confidence'] = {
                'distribution': np.histogram(all_confidences, bins=10)[0].tolist(),
                'mean': float(np.mean(all_confidences)),
                'std': float(np.std(all_confidences)),
                'quartiles': np.percentile(all_confidences, [25, 50, 75]).tolist()
            }
            
            # Anomaly summary
            all_anomalies = [a for r in results for a in r['anomalies']]
            summary['anomalies'] = {
                'total_count': len(all_anomalies),
                'types': self._summarize_anomaly_types(all_anomalies),
                'frequency': len(all_anomalies) / total_tests
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing test results: {str(e)}")
            return {}
    
    def _count_patterns(self, data: np.ndarray, window_size: int) -> float:
        """Count unique patterns in data"""
        patterns = set()
        for i in range(len(data) - window_size + 1):
            pattern = tuple(data[i:i+window_size])
            patterns.add(pattern)
        return len(patterns) / (len(data) - window_size + 1)
    
    def _analyze_detected_patterns(self, data: bytes, outputs: torch.Tensor) -> Dict[str, any]:
        """Analyze detected patterns in data"""
        patterns = {}
        
        # Convert outputs to pattern probabilities
        probs = torch.softmax(outputs, dim=1)[0]
        
        for i, pattern_class in enumerate(self.config.pattern_classes):
            if probs[i] > self.config.detection_threshold:
                patterns[pattern_class] = {
                    'probability': float(probs[i]),
                    'strength': self._calculate_pattern_strength(data, pattern_class),
                    'location': self._find_pattern_location(data, pattern_class)
                }
                
        return patterns
    
    def _detect_prediction_anomalies(self, 
                                   outputs: torch.Tensor,
                                   probabilities: torch.Tensor,
                                   data: bytes) -> List[Dict[str, any]]:
        """Detect anomalies in predictions"""
        anomalies = []
        
        # Check for low confidence
        if torch.max(probabilities) < self.config.confidence_threshold:
            anomalies.append({
                'type': 'low_confidence',
                'value': float(torch.max(probabilities)),
                'threshold': self.config.confidence_threshold
            })
            
        # Check for ambiguous predictions
        top2_probs, _ = torch.topk(probabilities, 2)
        if (top2_probs[0][0] - top2_probs[0][1]) < self.config.ambiguity_threshold:
            anomalies.append({
                'type': 'ambiguous_prediction',
                'margin': float(top2_probs[0][0] - top2_probs[0][1]),
                'threshold': self.config.ambiguity_threshold
            })
            
        return anomalies