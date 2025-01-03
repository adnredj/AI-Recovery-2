import time
from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging import setup_logger

class TrainingMonitor:
    """Monitor and track training progress"""
    
    def __init__(self, config: Dict[str, any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Initialize tracking
        self.metrics_history = {
            'train': [],
            'val': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Setup monitoring directory
        self.monitor_dir = Path(config.get('monitor_dir', 'monitoring'))
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize timing
        self.epoch_start_time = None
        self.training_start_time = time.time()
        
    def start_epoch(self):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()
        
    def end_epoch(self):
        """Mark the end of an epoch and record timing"""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.metrics_history['epoch_times'].append(epoch_time)
            self.epoch_start_time = None
            
    def log_metrics(self, 
                   epoch: int,
                   train_metrics: Dict[str, float],
                   val_metrics: Optional[Dict[str, float]] = None):
        """Log training and validation metrics
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        # Add epoch number to metrics
        train_metrics['epoch'] = epoch
        self.metrics_history['train'].append(train_metrics)
        
        if val_metrics:
            val_metrics['epoch'] = epoch
            self.metrics_history['val'].append(val_metrics)
            
        # Save metrics to file
        self._save_metrics()
        
        # Generate plots if configured
        if self.config.get('plot_metrics', True):
            self._generate_plots()
            
    def log_learning_rate(self, lr: float):
        """Log current learning rate
        
        Args:
            lr: Current learning rate
        """
        self.metrics_history['learning_rates'].append(lr)
        
    def get_summary(self) -> Dict[str, any]:
        """Get training summary
        
        Returns:
            Dictionary containing training summary
        """
        total_time = time.time() - self.training_start_time
        
        return {
            'total_time': total_time,
            'epochs_completed': len(self.metrics_history['train']),
            'average_epoch_time': np.mean(self.metrics_history['epoch_times']),
            'best_train_metrics': self._get_best_metrics('train'),
            'best_val_metrics': self._get_best_metrics('val'),
            'final_train_metrics': self.metrics_history['train'][-1] if self.metrics_history['train'] else None,
            'final_val_metrics': self.metrics_history['val'][-1] if self.metrics_history['val'] else None
        }
        
    def _save_metrics(self):
        """Save metrics history to file"""
        metrics_file = self.monitor_dir / 'metrics_history.json'
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            
    def _generate_plots(self):
        """Generate training progress plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = self.monitor_dir / 'plots' / timestamp
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss plot
        self._plot_metric('loss', plots_dir / 'loss.png')
        
        # Accuracy plots
        if 'accuracy' in self.metrics_history['train'][0]:
            self._plot_metric('accuracy', plots_dir / 'accuracy.png')
            
        # Learning rate plot
        if self.metrics_history['learning_rates']:
            self._plot_learning_rate(plots_dir / 'learning_rate.png')
            
        # Epoch time plot
        self._plot_epoch_times(plots_dir / 'epoch_times.png')
        
    def _plot_metric(self, metric_name: str, save_path: Path):
        """Plot specific metric over time
        
        Args:
            metric_name: Name of metric to plot
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics_history['train']) + 1)
        
        # Plot training metric
        train_values = [m[metric_name] for m in self.metrics_history['train']]
        plt.plot(epochs, train_values, 'b-', label=f'Train {metric_name}')
        
        # Plot validation metric if available
        if self.metrics_history['val']:
            val_values = [m[metric_name] for m in self.metrics_history['val']]
            plt.plot(epochs, val_values, 'r-', label=f'Val {metric_name}')
            
        plt.title(f'{metric_name.capitalize()} vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
    def _plot_learning_rate(self, save_path: Path):
        """Plot learning rate over time
        
        Args:
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics_history['learning_rates']) + 1)
        plt.plot(epochs, self.metrics_history['learning_rates'], 'g-')
        plt.title('Learning Rate vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
    def _plot_epoch_times(self, save_path: Path):
        """Plot epoch execution times
        
        Args:
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics_history['epoch_times']) + 1)
        plt.plot(epochs, self.metrics_history['epoch_times'], 'm-')
        plt.title('Epoch Time vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
    def _get_best_metrics(self, split: str) -> Optional[Dict[str, float]]:
        """Get best metrics for given split
        
        Args:
            split: Data split ('train' or 'val')
            
        Returns:
            Dictionary containing best metrics
        """
        if not self.metrics_history[split]:
            return None
            
        # Find best epoch based on loss
        best_epoch = min(
            range(len(self.metrics_history[split])),
            key=lambda i: self.metrics_history[split][i]['loss']
        )
        
        return self.metrics_history[split][best_epoch]