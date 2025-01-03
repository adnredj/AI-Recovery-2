import torch
import torch.nn as nn

class WalletEncoder(nn.Module):
    """Neural network encoder for wallet.dat analysis"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(1024)
        )
        
        # Cryptographic pattern detection
        self.crypto_detector = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=2,
            bidirectional=True
        )
        
        # Version-specific features
        self.version_encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

    def forward(self, x, version_info):
        # Extract low-level features
        features = self.feature_layers(x)
        
        # Process with LSTM for pattern detection
        crypto_features, _ = self.crypto_detector(features)
        
        # Encode version-specific information
        version_features = self.version_encoder(version_info)
        
        # Combine features
        combined = torch.cat([crypto_features, version_features], dim=-1)
        
        return combined