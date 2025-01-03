from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bitcoin
import hashlib

class RecoveryLoss:
    """Custom loss functions for bitcoin wallet recovery"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.entropy_weight = config.get('entropy_weight', 0.3)
        self.pattern_weight = config.get('pattern_weight', 0.3)
        self.consistency_weight = config.get('consistency_weight', 0.4)
        
    def combined_recovery_loss(self,
                             predictions: torch.Tensor,
                             targets: torch.Tensor,
                             pattern_matches: torch.Tensor,
                             entropy_scores: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss for recovery predictions"""
        # Basic prediction loss
        pred_loss = F.cross_entropy(predictions, targets)
        
        # Pattern matching loss
        pattern_loss = self._pattern_matching_loss(predictions, pattern_matches)
        
        # Entropy regularization
        entropy_loss = self._entropy_regularization(predictions, entropy_scores)
        
        # Consistency loss
        consistency_loss = self._consistency_loss(predictions)
        
        # Combine losses with weights
        total_loss = (pred_loss + 
                     self.pattern_weight * pattern_loss +
                     self.entropy_weight * entropy_loss +
                     self.consistency_weight * consistency_loss)
        
        return total_loss
    
    def _pattern_matching_loss(self,
                             predictions: torch.Tensor,
                             pattern_matches: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on pattern matching"""
        # Convert predictions to probabilities
        pred_probs = F.softmax(predictions, dim=-1)
        
        # Calculate pattern matching score
        pattern_scores = torch.sum(pred_probs * pattern_matches, dim=-1)
        
        # Convert to loss (higher pattern match = lower loss)
        loss = 1.0 - pattern_scores.mean()
        
        return loss
    
    def _entropy_regularization(self,
                              predictions: torch.Tensor,
                              target_entropy: torch.Tensor) -> torch.Tensor:
        """Entropy-based regularization loss"""
        # Calculate prediction entropy
        pred_probs = F.softmax(predictions, dim=-1)
        pred_entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=-1)
        
        # Calculate difference from target entropy
        entropy_diff = torch.abs(pred_entropy - target_entropy)
        
        return entropy_diff.mean()
    
    def _consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate consistency loss across predictions"""
        # Get probabilities
        pred_probs = F.softmax(predictions, dim=-1)
        
        # Calculate variance across batch
        pred_var = torch.var(pred_probs, dim=0)
        
        # Penalize high variance
        consistency_loss = pred_var.mean()
        
        return consistency_loss
    
    def key_validation_loss(self,
                          predicted_keys: torch.Tensor,
                          validation_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Loss function for key validation"""
        # Extract validation components
        addresses = validation_data['addresses']
        transactions = validation_data['transactions']
        signatures = validation_data['signatures']
        
        # Calculate individual validation losses
        address_loss = self._address_validation_loss(predicted_keys, addresses)
        transaction_loss = self._transaction_validation_loss(predicted_keys, transactions)
        signature_loss = self._signature_validation_loss(predicted_keys, signatures)
        
        # Combine validation losses
        total_loss = (address_loss + transaction_loss + signature_loss) / 3.0
        
        return total_loss
    
    def _address_validation_loss(self,
                               predicted_keys: torch.Tensor,
                               target_addresses: torch.Tensor) -> torch.Tensor:
        """Loss for address validation"""
        # Generate addresses from predicted keys
        generated_addresses = self._generate_addresses(predicted_keys)
        
        # Calculate address match loss
        match_loss = F.binary_cross_entropy(
            generated_addresses,
            target_addresses
        )
        
        return match_loss
    
    def _transaction_validation_loss(self,
                                   predicted_keys: torch.Tensor,
                                   transactions: torch.Tensor) -> torch.Tensor:
        """Loss for transaction validation"""
        # Verify transactions with predicted keys
        verification_scores = self._verify_transactions(predicted_keys, transactions)
        
        # Calculate verification loss
        verification_loss = 1.0 - verification_scores.mean()
        
        return verification_loss
    
    def _signature_validation_loss(self,
                                 predicted_keys: torch.Tensor,
                                 signatures: torch.Tensor) -> torch.Tensor:
        """Loss for signature validation"""
        # Verify signatures with predicted keys
        verification_scores = self._verify_signatures(predicted_keys, signatures)
        
        # Calculate signature verification loss
        verification_loss = 1.0 - verification_scores.mean()
        
        return verification_loss
    
    def _generate_addresses(self, keys: torch.Tensor) -> torch.Tensor:
        """Generate bitcoin addresses from keys
        
        Args:
            keys: Tensor of shape (batch_size, key_length) containing private keys
            
        Returns:
            Tensor of shape (batch_size, address_length) containing generated addresses
        """
        batch_size = keys.shape[0]
        addresses = torch.zeros((batch_size, 34), dtype=torch.uint8)  # Standard Bitcoin address length
        
        try:
            # Move to CPU for cryptographic operations
            keys_cpu = keys.cpu().numpy()
            
            for i in range(batch_size):
                # Convert key bytes to private key object
                private_key = bitcoin.SigningKey.from_string(
                    keys_cpu[i].tobytes(),
                    curve=bitcoin.SECP256k1
                )
                
                # Get public key
                public_key = private_key.get_verifying_key()
                
                # Generate address using public key
                address = bitcoin.pubkey_to_address(
                    public_key.to_string(),
                    version=0  # Mainnet version
                )
                
                # Convert address to tensor
                address_bytes = address.encode('ascii')
                addresses[i, :len(address_bytes)] = torch.tensor(
                    list(address_bytes), 
                    dtype=torch.uint8
                )
                
            return addresses.to(keys.device)
            
        except Exception as e:
            self.logger.error(f"Error generating addresses: {str(e)}")
            raise
    
    def _verify_transactions(self,
                            keys: torch.Tensor,
                            transactions: torch.Tensor) -> torch.Tensor:
        """Verify transactions with given keys
        
        Args:
            keys: Tensor of shape (batch_size, key_length) containing private keys
            transactions: Tensor of shape (batch_size, tx_data_length) containing transaction data
            
        Returns:
            Tensor of shape (batch_size,) containing verification results (1 for valid, 0 for invalid)
        """
        batch_size = keys.shape[0]
        results = torch.zeros(batch_size, dtype=torch.float32)
        
        try:
            # Move to CPU for cryptographic operations
            keys_cpu = keys.cpu().numpy()
            tx_cpu = transactions.cpu().numpy()
            
            for i in range(batch_size):
                try:
                    # Convert key bytes to private key object
                    private_key = bitcoin.SigningKey.from_string(
                        keys_cpu[i].tobytes(),
                        curve=bitcoin.SECP256k1
                    )
                    
                    # Extract transaction components
                    tx_hash = tx_cpu[i, :32]  # First 32 bytes are tx hash
                    signature = tx_cpu[i, 32:96]  # Next 64 bytes are signature
                    
                    # Verify transaction signature
                    public_key = private_key.get_verifying_key()
                    is_valid = public_key.verify(
                        signature,
                        tx_hash,
                        hashfunc=hashlib.sha256
                    )
                    
                    results[i] = float(is_valid)
                    
                except Exception as e:
                    self.logger.warning(f"Error verifying transaction {i}: {str(e)}")
                    results[i] = 0.0
                    
            return results.to(keys.device)
            
        except Exception as e:
            self.logger.error(f"Error in transaction verification: {str(e)}")
            raise
    
    def _verify_signatures(self,
                          keys: torch.Tensor,
                          signatures: torch.Tensor) -> torch.Tensor:
        """Verify signatures with given keys
        
        Args:
            keys: Tensor of shape (batch_size, key_length) containing private keys
            signatures: Tensor of shape (batch_size, signature_length) containing signatures
            
        Returns:
            Tensor of shape (batch_size,) containing verification results (1 for valid, 0 for invalid)
        """
        batch_size = keys.shape[0]
        results = torch.zeros(batch_size, dtype=torch.float32)
        
        try:
            # Move to CPU for cryptographic operations
            keys_cpu = keys.cpu().numpy()
            sig_cpu = signatures.cpu().numpy()
            
            for i in range(batch_size):
                try:
                    # Convert key bytes to private key object
                    private_key = bitcoin.SigningKey.from_string(
                        keys_cpu[i].tobytes(),
                        curve=bitcoin.SECP256k1
                    )
                    
                    # Extract signature components
                    message_hash = sig_cpu[i, :32]  # First 32 bytes are message hash
                    signature = sig_cpu[i, 32:96]  # Next 64 bytes are signature
                    
                    # Verify signature
                    public_key = private_key.get_verifying_key()
                    is_valid = public_key.verify(
                        signature,
                        message_hash,
                        hashfunc=hashlib.sha256
                    )
                    
                    results[i] = float(is_valid)
                    
                except Exception as e:
                    self.logger.warning(f"Error verifying signature {i}: {str(e)}")
                    results[i] = 0.0
                    
            return results.to(keys.device)
            
        except Exception as e:
            self.logger.error(f"Error in signature verification: {str(e)}")
            raise