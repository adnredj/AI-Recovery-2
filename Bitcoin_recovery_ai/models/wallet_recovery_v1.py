import torch
import torch.nn as nn
import yaml
from pathlib import Path

class WalletRecoveryModel(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        
        # Load model configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']
            
        # Initialize encoder
        self.encoder_layers = nn.ModuleList()
        prev_size = self.config['encoder']['input_size']
        
        for hidden_size in self.config['encoder']['hidden_sizes']:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config['encoder']['dropout'])
                )
            )
            prev_size = hidden_size
            
        # Initialize attention
        self.attention = nn.MultiheadAttention(
            embed_dim=prev_size,
            num_heads=self.config['attention']['num_heads'],
            dropout=self.config['attention']['dropout']
        )
        
        # Initialize decoder
        self.decoder_layers = nn.ModuleList()
        for hidden_size in self.config['decoder']['hidden_sizes']:
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config['decoder']['dropout'])
                )
            )
            prev_size = hidden_size
            
        # Output layers
        self.strategy_head = nn.Linear(prev_size, len(self.config['recovery_strategies']))
        self.confidence_head = nn.Linear(prev_size, 1)
        
    def forward(self, features):
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
        strategy_logits = self.strategy_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return {
            'strategy': torch.softmax(strategy_logits, dim=-1),
            'confidence': confidence
        }

def create_and_save_model():
    try:
        # Get config path
        config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
        
        # Create model
        model = WalletRecoveryModel(config_path)
        
        # Save path
        save_path = Path(__file__).parent / 'wallet_recovery_v1.pt'
        
        # Save model
        torch.save(model, save_path)
        print(f"Model saved successfully to {save_path}")
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")

if __name__ == '__main__':
    create_and_save_model()