from typing import Dict, List, Optional, Tuple, Union
import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidKey
import numpy as np

def verify_key_recovery(original_key: bytes, recovered_key: bytes) -> bool:
    """Verify if recovered key matches original
    
    Args:
        original_key: Original key bytes
        recovered_key: Recovered key bytes
        
    Returns:
        True if keys match
    """
    if not original_key or not recovered_key:
        return False
        
    try:
        # Compare key hashes
        original_hash = hashlib.sha256(original_key).digest()
        recovered_hash = hashlib.sha256(recovered_key).digest()
        return original_hash == recovered_hash
    except Exception:
        return False

class CryptoUtils:
    """Utility class for cryptographic operations"""
    
    @staticmethod
    def verify_key_recovery(original_key: bytes, recovered_key: bytes) -> bool:
        """Wrapper for global verify_key_recovery function"""
        return verify_key_recovery(original_key, recovered_key)
            
    @staticmethod
    def derive_key(password: str, salt: bytes, iterations: int = 600000) -> bytes:
        """Derive encryption key from password
        
        Args:
            password: Password string
            salt: Salt bytes
            iterations: Number of iterations
            
        Returns:
            Derived key bytes
        """
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations
            )
            return kdf.derive(password.encode())
        except Exception:
            return b''
            
    @staticmethod
    def decrypt_key(encrypted_key: bytes, 
                   key: bytes,
                   iv: Optional[bytes] = None) -> Optional[bytes]:
        """Decrypt an encrypted key
        
        Args:
            encrypted_key: Encrypted key bytes
            key: Decryption key
            iv: Initialization vector
            
        Returns:
            Decrypted key bytes or None if failed
        """
        if not encrypted_key or not key:
            return None
            
        try:
            if iv:
                # AES-CBC decryption
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CBC(iv)
                )
            else:
                # AES-ECB decryption
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.ECB()
                )
                
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(encrypted_key) + decryptor.finalize()
            
            # Remove PKCS7 padding
            padding_len = decrypted[-1]
            if padding_len > 16:
                return None
            return decrypted[:-padding_len]
            
        except Exception:
            return None
            
    @staticmethod
    def analyze_key_pattern(key_data: bytes) -> Dict[str, float]:
        """Analyze encryption patterns in key data
        
        Args:
            key_data: Key data bytes
            
        Returns:
            Dictionary of pattern probabilities
        """
        if not key_data or len(key_data) < 16:
            return {'unknown': 1.0}
            
        try:
            # Calculate entropy
            entropy = CryptoUtils._calculate_entropy(key_data)
            
            # Analyze block patterns
            patterns = {
                'aes': CryptoUtils._check_aes_pattern(key_data),
                'chacha20': CryptoUtils._check_chacha_pattern(key_data),
                'unknown': 0.0
            }
            
            # Weight by entropy
            entropy_weight = min(entropy / 8.0, 1.0)
            for k in patterns:
                if k != 'unknown':
                    patterns[k] *= entropy_weight
                    
            # Fill unknown probability
            total_prob = sum(v for k, v in patterns.items() if k != 'unknown')
            patterns['unknown'] = 1.0 - total_prob
            
            return patterns
            
        except Exception:
            return {'unknown': 1.0}
            
    @staticmethod
    def _calculate_entropy(data: bytes) -> float:
        """Calculate Shannon entropy of data
        
        Args:
            data: Input bytes
            
        Returns:
            Entropy value
        """
        if not data:
            return 0.0
            
        # Calculate byte frequencies
        freq = np.zeros(256)
        for byte in data:
            freq[byte] += 1
            
        # Convert to probabilities
        prob = freq / len(data)
        
        # Calculate entropy
        entropy = 0.0
        for p in prob:
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
        
    @staticmethod
    def _check_aes_pattern(data: bytes) -> float:
        """Check for AES encryption patterns
        
        Args:
            data: Input bytes
            
        Returns:
            Pattern match probability
        """
        score = 0.0
        
        # Check block size
        if len(data) % 16 == 0:
            score += 0.3
            
        # Check for common AES patterns
        if len(data) >= 16:
            header = data[:16]
            if header[0] in (0x00, 0x01):  # Common AES headers
                score += 0.2
            if sum(header[1:4]) > 0:  # Non-zero IV indicator
                score += 0.2
            if header[-4:] != b'\x00' * 4:  # Non-zero padding
                score += 0.3
                
        return min(score, 1.0)
        
    @staticmethod
    def _check_chacha_pattern(data: bytes) -> float:
        """Check for ChaCha20 encryption patterns
        
        Args:
            data: Input bytes
            
        Returns:
            Pattern match probability
        """
        score = 0.0
        
        # Check block size
        if len(data) % 64 == 0:
            score += 0.4
            
        # Check for ChaCha20 patterns
        if len(data) >= 16:
            header = data[:16]
            if header[0] == 0x01 and header[1] == 0x02:  # Common ChaCha header
                score += 0.3
            if sum(header[2:6]) > 0:  # Non-zero counter
                score += 0.3
                
        return min(score, 1.0)

# Export the function at module level
__all__ = ['verify_key_recovery', 'CryptoUtils']