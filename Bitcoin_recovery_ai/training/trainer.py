import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from bitcoin_recovery_ai.utils.logging import setup_logger
from bitcoin_recovery_ai.utils.metrics import calculate_recovery_metrics
from bitcoin_recovery_ai.monitoring.training_monitor import TrainingMonitor
from bitcoin_recovery_ai.data.loader import create_train_loader, create_test_loader

# Make wandb optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class ModelTrainer:
    """Trainer for wallet recovery model"""
    
    def __init__(self, 
                 config: Dict[str, any],
                 device: Optional[torch.device] = None):
        """
        Args:
            config: Training configuration
            device: Device to train on
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger(__name__)
        
        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize monitor
        self.monitor = TrainingMonitor(config)
        
        # Loss functions with weights
        self.recovery_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config.get('recovery_pos_weight', 1.0)]).to(self.device)
        )
        self.pattern_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(config.get('pattern_weights', [1.0])).to(self.device)
        )
        self.entropy_loss = nn.MSELoss()
        
        # Initialize tracking
        self.best_metrics = {
            'recovery_acc': 0.0,
            'pattern_acc': 0.0,
            'total_loss': float('inf')
        }
        self.current_epoch = 0
        
        # Initialize wandb if enabled and available
        if config.get('use_wandb', False):
            if WANDB_AVAILABLE:
                self._init_wandb()
            else:
                self.logger.warning("WandB logging disabled - package not installed")
                self.config['use_wandb'] = False
                
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'transformer')
        
        if model_type == 'transformer':
            from bitcoin_recovery_ai.models.transformer import WalletTransformer
            return WalletTransformer(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        optimizer_config = self.config.get('training', {}).get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('learning_rate', 1e-4)
        
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        num_epochs = self.config.get('training', {}).get('epochs', 100)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase if loader provided
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self._log_metrics({**train_metrics, **{'val_' + k: v for k, v in val_metrics.items()}})
            
            # Check for early stopping
            if self._should_stop_early(train_metrics):
                self.logger.info("Early stopping triggered")
                break

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary containing epoch metrics
        """
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'recovery_loss': 0.0,
            'pattern_loss': 0.0,
            'entropy_loss': 0.0,
            'recovery_acc': 0.0,
            'pattern_acc': 0.0,
            'entropy_rmse': 0.0
        }
        
        with tqdm(dataloader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch in pbar:
                # Move data to device
                wallet_data = batch['data'].to(self.device)
                version_info = batch['version'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(wallet_data, version_info)
                
                # Calculate individual losses
                recovery_loss = self.recovery_loss(
                    outputs['recovery_probability'],
                    targets['recovery']
                )
                pattern_loss = self.pattern_loss(
                    outputs['encryption_pattern'],
                    targets['pattern']
                )
                entropy_loss = self.entropy_loss(
                    outputs['entropy'],
                    targets['entropy']
                )
                
                # Combined loss with weights
                total_loss = (
                    self.config.recovery_weight * recovery_loss +
                    self.config.pattern_weight * pattern_loss +
                    self.config.entropy_weight * entropy_loss
                )
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping if configured
                if self.config.get('clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['clip_grad_norm']
                    )
                    
                self.optimizer.step()
                
                # Update batch metrics
                batch_metrics = {
                    'total_loss': total_loss.item(),
                    'recovery_loss': recovery_loss.item(),
                    'pattern_loss': pattern_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'recovery_acc': self._calculate_recovery_accuracy(
                        outputs['recovery_probability'],
                        targets['recovery']
                    ),
                    'pattern_acc': self._calculate_pattern_accuracy(
                        outputs['encryption_pattern'],
                        targets['pattern']
                    ),
                    'entropy_rmse': torch.sqrt(entropy_loss).item()
                }
                
                # Update epoch metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                    
                # Update progress bar
                pbar.set_postfix({
                    'loss': batch_metrics['total_loss'],
                    'rec_acc': batch_metrics['recovery_acc']
                })
                
        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)
            
        # Log metrics
        self._log_metrics(epoch_metrics)
        
        # Save checkpoint if improved
        if self._is_best_model(epoch_metrics):
            self._save_checkpoint(epoch_metrics)
            
        self.current_epoch += 1
        return epoch_metrics
    
    def _calculate_recovery_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate recovery prediction accuracy
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Accuracy score
        """
        with torch.no_grad():
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (pred_binary == targets).float().mean().item()
            
            # Calculate additional metrics if needed
            if self.config.get('detailed_metrics', False):
                metrics = calculate_recovery_metrics(pred_binary.cpu(), targets.cpu())
                return metrics['accuracy']
                
            return accuracy
    
    def _calculate_pattern_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate pattern detection accuracy
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Accuracy score
        """
        with torch.no_grad():
            pred_classes = torch.argmax(predictions, dim=1)
            return (pred_classes == targets).float().mean().item()
            
    def _is_best_model(self, metrics: Dict[str, float]) -> bool:
        """Check if current model is best so far
        
        Args:
            metrics: Current epoch metrics
            
        Returns:
            True if model is best so far
        """
        # Define improvement criteria
        improvements = {
            'recovery_acc': metrics['recovery_acc'] > self.best_metrics['recovery_acc'],
            'pattern_acc': metrics['pattern_acc'] > self.best_metrics['pattern_acc'],
            'total_loss': metrics['total_loss'] < self.best_metrics['total_loss']
        }
        
        # Update best metrics if improved
        if any(improvements.values()):
            self.best_metrics.update(metrics)
            return True
            
        return False
        
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint
        
        Args:
            metrics: Current epoch metrics
        """
        checkpoint_path = Path(self.config['checkpoint_dir']) / f'model_epoch_{self.current_epoch}.pt'
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metrics': self.best_metrics,
            'config': self.config
        }, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics
        
        Args:
            metrics: Metrics to log
        """
        # Console logging
        self.logger.info(
            f"Epoch {self.current_epoch} - "
            f"Loss: {metrics['total_loss']:.4f}, "
            f"Recovery Acc: {metrics['recovery_acc']:.4f}, "
            f"Pattern Acc: {metrics['pattern_acc']:.4f}"
        )
        
        # WandB logging if enabled and available
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log({
                'epoch': self.current_epoch,
                **metrics
            })
            
        # Monitor logging
        self.monitor.log_metrics(self.current_epoch, metrics)

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        if WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get('wandb_project', 'wallet-recovery'),
                config=self.config,
                name=self.config.get('wandb_run_name'),
                group=self.config.get('wandb_group')
            )