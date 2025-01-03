import torch
from typing import Dict, Optional, List
from ..utils.logging import setup_logger
import os
from ..models.model import WalletRecoveryModel

class RecoveryValidator:
    """Validator for wallet recovery attempts"""
    
    def __init__(self, config: Dict[str, any], device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger(__name__)
        
    def recover(self, features: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Attempt wallet recovery using extracted features"""
        try:
            # Process and combine features
            combined_features = self._combine_features(features)
            
            # Load model
            model = self._load_model()
            model.eval()
            
            # Get predictions
            with torch.no_grad():
                predictions = model(combined_features)
                
            # Process predictions
            results = self._process_predictions(predictions)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _combine_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine different feature types into a single tensor"""
        try:
            # Fixed feature dimensions
            feature_dims = {
                'structure': 64,
                'version': 32,
                'transaction': 48,
                'address': 32,
                'metadata': 16
            }
            
            # Process each feature type in order
            processed_features = []
            for feature_name, expected_dim in feature_dims.items():
                if feature_name in features:
                    feature = features[feature_name]
                    
                    # Convert to tensor if needed
                    if isinstance(feature, list):
                        feature = torch.tensor(feature, dtype=torch.float32)
                    
                    # Ensure 1D
                    if feature.dim() > 1:
                        feature = feature.flatten()
                        
                    # Ensure correct dimension
                    if len(feature) < expected_dim:
                        # Pad with zeros
                        feature = torch.cat([
                            feature,
                            torch.zeros(expected_dim - len(feature), dtype=torch.float32)
                        ])
                    else:
                        # Truncate if too long
                        feature = feature[:expected_dim]
                else:
                    # Create zero tensor if feature is missing
                    feature = torch.zeros(expected_dim, dtype=torch.float32)
                    
                processed_features.append(feature)
                    
            # Combine all features
            combined = torch.cat(processed_features)
            
            # Add batch dimension and move to device
            combined = combined.unsqueeze(0).to(self.device)
            
            self.logger.debug(f"Combined feature shape: {combined.shape}")
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining features: {str(e)}")
            raise RuntimeError(f"Feature combination failed: {str(e)}")
            
    def _load_model(self) -> torch.nn.Module:
        """Load the recovery model"""
        try:
            # Get model path from config
            model_path = self.config.get('model', {}).get('path')
            if not model_path:
                raise ValueError("Model path not specified in config")
                
            # Convert to absolute path if needed
            model_path = os.path.abspath(model_path)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
            self.logger.info(f"Loading model from: {model_path}")
            
            # Load model state dict with weights_only=True
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Create new model instance
            model = WalletRecoveryModel()
            
            # Load state dict
            model.load_state_dict(state_dict)
            
            # Move model to device
            model = model.to(self.device)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    def _process_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Process model predictions
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Processed results dictionary
        """
        results = {
            'success': True,
            'confidence': float(predictions.get('confidence', 0.0)),
            'recovery_type': self._get_recovery_type(predictions),
            'encryption_info': self._get_encryption_info(predictions),
            'recommendations': self._get_recommendations(predictions)
        }
        
        return results
        
    def _get_recovery_type(self, predictions: Dict[str, torch.Tensor]) -> str:
        """Determine recovery type from predictions"""
        try:
            # Get recovery type probabilities and encryption probability
            type_probs = predictions.get('recovery_type', torch.zeros(4))[0]
            enc_prob = predictions.get('encryption_prob', torch.zeros(1))[0].item()
            
            # Map indices to recovery types
            recovery_types = {
                0: 'standard',
                1: 'encrypted',
                2: 'corrupted',
                3: 'hardware'
            }
            
            # First check encryption probability
            if enc_prob > 0.6:  # Higher threshold for encryption detection
                return 'encrypted'
                
            # Get highest probability type
            type_idx = type_probs.argmax().item()
            recovery_type = recovery_types.get(type_idx, 'unknown')
            
            # Check confidence threshold
            confidence = type_probs[type_idx].item()
            if confidence < self.config.get('type_confidence_threshold', 0.7):
                self.logger.warning(f"Low confidence ({confidence:.2f}) in recovery type prediction")
                
                # Additional checks for encrypted wallets
                if 'encryption_type' in predictions:
                    enc_type_probs = predictions['encryption_type'][0]
                    if enc_type_probs.max().item() > 0.4:  # Lower threshold for secondary check
                        return 'encrypted'
                        
                # Check for encryption patterns in features
                if 'auth_requirements' in predictions:
                    auth_probs = predictions['auth_requirements'][0]
                    if auth_probs[0].item() > 0.4:  # Password requirement indicator
                        return 'encrypted'
            
            return recovery_type
            
        except Exception as e:
            self.logger.error(f"Error determining recovery type: {str(e)}")
            return 'unknown'
            
    def _get_encryption_info(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Extract encryption information from predictions"""
        try:
            encryption_info = {
                'is_encrypted': False,
                'encryption_type': 'none',
                'encryption_version': None,
                'encryption_strength': 0,
                'estimated_complexity': 'low',
                'requires_password': False,
                'requires_key_file': False,
                'additional_params': {}
            }
            
            # Check if encrypted
            if 'encryption_prob' in predictions:
                enc_prob = predictions['encryption_prob'][0].item()
                encryption_info['is_encrypted'] = enc_prob > 0.5
                
            # Get encryption type if encrypted
            if encryption_info['is_encrypted'] and 'encryption_type' in predictions:
                type_probs = predictions['encryption_type'][0]
                enc_types = {
                    0: 'aes-256-cbc',
                    1: 'chacha20-poly1305',
                    2: 'legacy-bitcoin-core',
                    3: 'custom'
                }
                type_idx = type_probs.argmax().item()
                encryption_info['encryption_type'] = enc_types.get(type_idx, 'unknown')
                
            # Check authentication requirements
            if 'auth_requirements' in predictions:
                auth_probs = predictions['auth_requirements'][0]
                encryption_info['requires_password'] = auth_probs[0].item() > 0.5
                encryption_info['requires_key_file'] = auth_probs[1].item() > 0.5
                
            return encryption_info
            
        except Exception as e:
            self.logger.error(f"Error extracting encryption info: {str(e)}")
            return {'is_encrypted': False, 'encryption_type': 'unknown'}
            
    def _get_recommendations(self, predictions: Dict[str, torch.Tensor]) -> List[str]:
        """Generate recovery recommendations based on predictions"""
        try:
            recommendations = []
            
            # Get recovery type and encryption info
            recovery_type = self._get_recovery_type(predictions)
            encryption_info = self._get_encryption_info(predictions)
            
            # Basic recommendations based on recovery type
            if recovery_type == 'standard':
                recommendations.append("Use standard Bitcoin Core recovery process")
            elif recovery_type == 'encrypted':
                recommendations.append("Wallet is encrypted - password will be required")
            elif recovery_type == 'corrupted':
                recommendations.append("Wallet appears corrupted - specialized recovery tools needed")
            elif recovery_type == 'hardware':
                recommendations.append("Connect corresponding hardware wallet device")
                
            # Encryption-specific recommendations
            if encryption_info['is_encrypted']:
                if encryption_info['encryption_type'] == 'legacy-bitcoin-core':
                    recommendations.append("Use Bitcoin Core version 0.7 or later for recovery")
                if encryption_info['requires_password']:
                    recommendations.append("Original wallet password required")
                if encryption_info['requires_key_file']:
                    recommendations.append("Key file needed in addition to password")
                    
            # Add performance recommendations
            if 'performance_requirements' in predictions:
                perf = predictions['performance_requirements'][0]
                if perf[0].item() > 0.5:  # GPU recommended
                    recommendations.append("GPU acceleration recommended for faster recovery")
                if perf[1].item() > 0.5:  # High memory
                    recommendations.append("At least 16GB RAM recommended")
                if perf[2].item() > 0.5:  # Multi-threading
                    recommendations.append("Multi-core CPU recommended")
                    
            # Add backup recommendations
            recommendations.append("Create backup of wallet file before recovery attempt")
            recommendations.append("Use secure, isolated environment for recovery")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to error"]