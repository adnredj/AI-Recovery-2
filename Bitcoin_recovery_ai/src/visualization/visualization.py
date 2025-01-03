import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

class RecoveryVisualizer:
    """Visualization tools for wallet recovery analysis"""
    
    def __init__(self, config):
        self.config = config
        self.plot_dir = Path('plots')
        self.plot_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def plot_recovery_metrics(self,
                            metrics: Dict[str, List[float]],
                            save_path: Optional[str] = None):
        """Plot recovery metrics over time"""
        plt.figure(figsize=(12, 6))
        
        # Plot training metrics
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['val_loss'], label='Validation Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy metrics
        plt.subplot(1, 2, 2)
        plt.plot(metrics['recovery_accuracy'], label='Recovery Accuracy')
        plt.plot(metrics['pattern_accuracy'], label='Pattern Detection Accuracy')
        plt.title('Accuracy Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_confidence_distribution(self,
                                  predictions: Dict[str, float],
                                  save_path: Optional[str] = None):
        """Plot confidence score distributions"""
        plt.figure(figsize=(10, 6))
        
        # Create confidence distribution plot
        sns.kdeplot(predictions.values(), fill=True)
        plt.axvline(x=self.config.thresholds.confidence_threshold, 
                   color='r', linestyle='--', label='Confidence Threshold')
        
        plt.title('Recovery Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_pattern_analysis(self,
                            pattern_data: Dict[str, any],
                            save_path: Optional[str] = None):
        """Plot pattern analysis results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Pattern Distribution
        if 'patterns' in pattern_data:
            patterns = pattern_data['patterns']
            names = list(patterns.keys())
            values = list(patterns.values())
            sns.barplot(x=values, y=names, ax=ax1)
            ax1.set_title('Pattern Distribution')
            ax1.set_xlabel('Frequency')
            
        # 2. Pattern Confidence
        if 'confidence_scores' in pattern_data:
            scores = pattern_data['confidence_scores']
            sns.heatmap(scores, 
                       ax=ax2, 
                       cmap='YlOrRd',
                       annot=True,
                       fmt='.2f')
            ax2.set_title('Pattern Confidence')
            
        # 3. Time Series
        if 'time_series' in pattern_data:
            time_data = pattern_data['time_series']
            ax3.plot(time_data['timestamps'], 
                    time_data['values'],
                    marker='o')
            ax3.set_title('Pattern Evolution')
            ax3.tick_params(axis='x', rotation=45)
            
        # 4. Pattern Correlations
        if 'correlations' in pattern_data:
            corr_matrix = pattern_data['correlations']
            sns.heatmap(corr_matrix,
                       ax=ax4,
                       cmap='coolwarm',
                       center=0,
                       annot=True,
                       fmt='.2f')
            ax4.set_title('Pattern Correlations')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_encryption_analysis(self,
                               encryption_data: Dict[str, any],
                               save_path: Optional[str] = None):
        """Plot encryption analysis results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Encryption Methods Distribution
        if 'methods' in encryption_data:
            methods = encryption_data['methods']
            ax1.pie(methods.values(),
                   labels=methods.keys(),
                   autopct='%1.1f%%')
            ax1.set_title('Encryption Methods')
            
        # 2. Strength Analysis
        if 'strength' in encryption_data:
            strength = encryption_data['strength']
            categories = list(strength.keys())
            values = list(strength.values())
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            values = np.concatenate((values, [values[0]]))  # complete the circle
            angles = np.concatenate((angles, [angles[0]]))  # complete the circle
            
            ax2.plot(angles, values)
            ax2.fill(angles, values, alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories)
            ax2.set_title('Encryption Strength')
            
        # 3. Vulnerability Analysis
        if 'vulnerabilities' in encryption_data:
            vulns = encryption_data['vulnerabilities']
            names = list(vulns.keys())
            scores = list(vulns.values())
            
            sns.barplot(x=scores, y=names, ax=ax3)
            ax3.set_title('Vulnerability Analysis')
            ax3.set_xlabel('Risk Score')
            
        # 4. Key Derivation Analysis
        if 'key_derivation' in encryption_data:
            kd_data = encryption_data['key_derivation']
            if 'iterations' in kd_data and 'strength' in kd_data:
                ax4.scatter(kd_data['iterations'],
                          kd_data['strength'])
                ax4.set_title('Key Derivation Analysis')
                ax4.set_xlabel('Iterations')
                ax4.set_ylabel('Strength Score')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()