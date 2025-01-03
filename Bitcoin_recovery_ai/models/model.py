import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class BaseModel(nn.Module):
    """Base class for all wallet recovery models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model"""
        raise NotImplementedError("Subclasses must implement forward()")
        
    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.config = checkpoint['config']

class WalletRecoveryModel(BaseModel):
    """Main wallet recovery model implementation"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)['model']
        else:
            config = {
                'encoder': {
                    'input_size': 192,  # Sum of feature dimensions
                    'hidden_sizes': [256, 512],
                    'dropout': 0.3
                },
                'attention': {
                    'num_heads': 8,
                    'dropout': 0.1
                },
                'decoder': {
                    'hidden_sizes': [512, 256, 128],
                    'dropout': 0.3
                }
            }
            
        super().__init__(config)
        
        # Initialize encoder
        self.encoder_layers = self._build_encoder()
        
        # Initialize attention
        encoder_output_size = self.config['encoder']['hidden_sizes'][-1]
        self.attention = nn.MultiheadAttention(
            embed_dim=encoder_output_size,
            num_heads=self.config['attention']['num_heads'],
            dropout=self.config['attention']['dropout']
        )
        
        # Initialize decoder
        self.decoder_layers = self._build_decoder()
        
        # Initialize output heads
        self._build_output_heads()
        
    def _build_encoder(self) -> nn.ModuleList:
        """Build encoder layers"""
        layers = nn.ModuleList()
        prev_size = self.config['encoder']['input_size']
        
        for hidden_size in self.config['encoder']['hidden_sizes']:
            layers.append(
                nn.Sequential(
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config['encoder']['dropout'])
                )
            )
            prev_size = hidden_size
            
        return layers
        
    def _build_decoder(self) -> nn.ModuleList:
        """Build decoder layers"""
        layers = nn.ModuleList()
        prev_size = self.config['encoder']['hidden_sizes'][-1]
        
        for hidden_size in self.config['decoder']['hidden_sizes']:
            layers.append(
                nn.Sequential(
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config['decoder']['dropout'])
                )
            )
            prev_size = hidden_size
            
        return layers
        
    def _build_output_heads(self):
        """Initialize output heads"""
        final_size = self.config['decoder']['hidden_sizes'][-1]
        
        self.recovery_type_head = nn.Linear(final_size, 4)  # 4 recovery types
        self.encryption_type_head = nn.Linear(final_size, 4)  # 4 encryption types
        self.encryption_prob_head = nn.Linear(final_size, 1)  # Encryption probability
        self.confidence_head = nn.Linear(final_size, 1)  # Overall confidence
        self.auth_head = nn.Linear(final_size, 2)  # Password/keyfile requirements
        self.performance_head = nn.Linear(final_size, 3)  # Performance requirements
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model
        
        Args:
            features: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Dictionary containing model predictions
            
        Raises:
            RuntimeError: If there's an error during forward pass
        """
        try:
            # Encode features
            x = features
            for layer in self.encoder_layers:
                x = layer(x)
                
            # Apply attention
            x = x.unsqueeze(0)  # Add sequence dimension
            attn_output, _ = self.attention(x, x, x)
            x = attn_output.squeeze(0)  # Remove sequence dimension
            
            # Decode
            for layer in self.decoder_layers:
                x = layer(x)
                
            # Generate outputs
            return {
                'recovery_type': torch.softmax(self.recovery_type_head(x), dim=-1),
                'encryption_type': torch.softmax(self.encryption_type_head(x), dim=-1),
                'encryption_prob': torch.sigmoid(self.encryption_prob_head(x)),
                'confidence': torch.sigmoid(self.confidence_head(x)),
                'auth_requirements': torch.sigmoid(self.auth_head(x)),
                'performance_requirements': torch.sigmoid(self.performance_head(x))
            }
            
        except Exception as e:
            raise RuntimeError(f"Forward pass error: {str(e)}") from e

def create_and_save_model(save_path: Optional[str] = None) -> WalletRecoveryModel:
    """Create and save a new model instance
    
    Args:
        save_path: Optional path to save the model. If None, saves to default location.
        
    Returns:
        Initialized model instance
        
    Raises:
        RuntimeError: If there's an error creating or saving the model
    """
    try:
        # Create model
        model = WalletRecoveryModel()
        
        # Determine save path
        if save_path is None:
            save_path = Path(__file__).parent / 'wallet_recovery_v1.pt'
        
        # Save model
        model.save(str(save_path))
        print(f"Model saved successfully to {save_path}")
        
        # Test forward pass
        test_input = torch.randn(1, 192)  # Test with correct input size
        with torch.no_grad():
            output = model(test_input)
            print("Test forward pass successful")
            print(f"Output shapes: {[(k, v.shape) for k, v in output.items()]}")
            
        return model
            
    except Exception as e:
        raise RuntimeError(f"Error creating model: {str(e)}") from e

if __name__ == '__main__':
    create_and_save_model()