import torch
from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader
from ..utils.logging import setup_logger
from ..utils.training_utils import TrainingUtils
from ..models.components.crypto_analyzer import CryptoAnalyzer
from ..validation.validator import WalletRecoveryValidator

class WalletRecoveryTrainer:
    """Main training loop for wallet recovery model"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = setup_logger(__name__)
        self.training_utils = TrainingUtils(config)
        self.validator = WalletRecoveryValidator(config, device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.training.lr_scheduler.patience,
            factor=config.training.lr_scheduler.factor,
            min_lr=config.training.lr_scheduler.min_lr
        )
        
        # Setup checkpointing
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int) -> Dict[str, list]:
        """Main training loop"""
        history = {
            'train_loss': [],
            'val_loss': [],
            'recovery_accuracy': [],
            'pattern_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            history['train_loss'].append(train_metrics['total_loss'])
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            history['val_loss'].append(val_metrics['total_loss'])
            history['recovery_accuracy'].append(val_metrics['recovery_accuracy'])
            history['pattern_accuracy'].append(val_metrics['pattern_accuracy'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Checkpointing
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self._save_checkpoint(epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.training.params.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Logging
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {'total_loss': 0.0}
        
        for batch_idx, batch in enumerate(dataloader):
            # Prepare batch
            inputs, targets = self.training_utils.prepare_batch(batch, self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(**inputs)
            
            # Calculate loss
            loss, batch_metrics = self.training_utils.calculate_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.params.gradient_clip_val
            )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['total_loss'] += loss.item()
            
            # Log batch metrics
            if batch_idx % self.config.training.logging.log_interval == 0:
                self._log_batch_metrics(batch_idx, len(dataloader), batch_metrics)
                
        # Average metrics
        epoch_metrics['total_loss'] /= len(dataloader)
        return epoch_metrics
    
    def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'recovery_accuracy': 0.0,
            'pattern_accuracy': 0.0
        }
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self.training_utils.prepare_batch(batch, self.device)
                predictions = self.model(**inputs)
                
                # Calculate metrics
                _, batch_metrics = self.training_utils.calculate_loss(predictions, targets)
                val_metrics = self.training_utils.update_metrics(val_metrics, predictions, targets)
                val_metrics['total_loss'] += batch_metrics['total_loss']
                
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= len(dataloader)
            
        return val_metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _log_batch_metrics(self, batch_idx: int, num_batches: int, metrics: Dict[str, float]):
        """Log batch-level metrics"""
        message = f"Batch [{batch_idx}/{num_batches}] "
        message += " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(message)
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        message = f"\nEpoch {epoch}:\n"
        message += "Training: " + " ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]) + "\n"
        message += "Validation: " + " ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        self.logger.info(message)