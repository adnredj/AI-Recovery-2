import torch
from pathlib import Path
from .model import WalletRecoveryModel

def create_and_save_model():
    try:
        # Get config path
        config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
        
        # Create model
        model = WalletRecoveryModel(config_path)
        
        # Save path
        save_path = Path(__file__).parent / 'wallet_recovery_v1.pt'
        
        # Save model state dict instead of full model
        torch.save(model.state_dict(), save_path)
        print(f"Model saved successfully to {save_path}")
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")

if __name__ == '__main__':
    create_and_save_model()