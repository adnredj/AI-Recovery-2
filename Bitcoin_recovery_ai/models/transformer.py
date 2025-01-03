import torch
import torch.nn as nn
import math
from typing import Dict, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create position encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class WalletTransformer(nn.Module):
    """Transformer model for wallet recovery"""
    
    def __init__(self, config: Dict[str, any]):
        super().__init__()
        
        # Get feature dimensions from config
        feature_config = config.get('features', {})
        self.structure_dim = feature_config.get('structure_dim', 64)
        self.version_dim = feature_config.get('version_dim', 32)
        self.transaction_dim = feature_config.get('transaction_dim', 48)
        self.address_dim = feature_config.get('address_dim', 32)
        self.metadata_dim = feature_config.get('metadata_dim', 16)
        
        # Calculate total input dimension
        self.input_dim = (self.structure_dim + self.version_dim + 
                         self.transaction_dim + self.address_dim + 
                         self.metadata_dim)
        
        # Get model configuration
        model_config = config.get('model', {})
        self.d_model = model_config.get('embedding_dim', 256)
        self.nhead = model_config.get('num_heads', 8)
        self.num_layers = model_config.get('num_layers', 6)
        self.dropout = model_config.get('dropout', 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output heads
        self.recovery_type_head = nn.Linear(self.d_model, 4)
        self.encryption_type_head = nn.Linear(self.d_model, 4)
        self.encryption_prob_head = nn.Linear(self.d_model, 1)
        self.confidence_head = nn.Linear(self.d_model, 1)
        self.auth_head = nn.Linear(self.d_model, 2)
        self.performance_head = nn.Linear(self.d_model, 3)
        
    def _combine_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine different feature types into a single tensor"""
        try:
            feature_list = []
            
            # Expected feature types and dimensions
            feature_dims = {
                'structure': self.structure_dim,
                'version': self.version_dim,
                'transaction': self.transaction_dim,
                'address': self.address_dim,
                'metadata': self.metadata_dim
            }
            
            # Process each feature type
            for feature_name, expected_dim in feature_dims.items():
                if feature_name in features:
                    feature = features[feature_name]
                    
                    # Convert to tensor if needed
                    if not isinstance(feature, torch.Tensor):
                        feature = torch.tensor(feature, dtype=torch.float32)
                        
                    # Ensure correct shape
                    if feature.dim() == 1:
                        feature = feature[:expected_dim]
                        if len(feature) < expected_dim:
                            padding = torch.zeros(expected_dim - len(feature), 
                                               dtype=torch.float32)
                            feature = torch.cat([feature, padding])
                    
                    feature_list.append(feature)
                else:
                    # Create zero tensor if feature is missing
                    feature_list.append(torch.zeros(expected_dim, dtype=torch.float32))
                    
            # Combine all features
            return torch.cat(feature_list)
            
        except Exception as e:
            print(f"Feature combination error: {str(e)}")
            raise e
            
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            # Print debug info
            print(f"Input features: {[k for k in features.keys()]}")
            print(f"Feature shapes: {[(k, v.shape) for k, v in features.items()]}")
            
            # Combine features
            combined_features = self._combine_features(features)
            print(f"Combined features shape: {combined_features.shape}")
            
            # Project input to d_model dimensions
            x = self.input_projection(combined_features)
            
            # Add batch dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(0)
                
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Apply transformer encoder
            x = self.transformer_encoder(x)
            
            # Global average pooling
            x = x.mean(dim=1)
            
            # Generate predictions
            return {
                'recovery_type': torch.softmax(self.recovery_type_head(x), dim=-1),
                'encryption_type': torch.softmax(self.encryption_type_head(x), dim=-1),
                'encryption_prob': torch.sigmoid(self.encryption_prob_head(x)),
                'confidence': torch.sigmoid(self.confidence_head(x)),
                'auth_requirements': torch.sigmoid(self.auth_head(x)),
                'performance_requirements': torch.sigmoid(self.performance_head(x))
            }
            
        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            raise e

class TransformerTrainer:
    def __init__(self, config: Dict[str, any], device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = WalletTransformer(config).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0)
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(batch['features'])
        
        # Calculate losses
        losses = {}
        total_loss = 0.0
        
        # Recovery type loss
        if 'recovery_type' in batch:
            loss = nn.CrossEntropyLoss()(
                outputs['recovery_type'],
                batch['recovery_type']
            )
            losses['recovery_type'] = loss.item()
            total_loss += loss
            
        # Encryption type loss
        if 'encryption_type' in batch:
            loss = nn.CrossEntropyLoss()(
                outputs['encryption_type'],
                batch['encryption_type']
            )
            losses['encryption_type'] = loss.item()
            total_loss += loss
            
        # Binary losses
        binary_targets = ['encryption_prob', 'auth_requirements', 'performance_requirements']
        for target in binary_targets:
            if target in batch:
                loss = nn.BCELoss()(
                    outputs[target],
                    batch[target].float()
                )
                losses[target] = loss.item()
                total_loss += loss
                
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        losses['total'] = total_loss.item()
        return losses
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])