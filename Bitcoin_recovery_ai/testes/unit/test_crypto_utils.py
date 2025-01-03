import unittest
import numpy as np
from ..src.utils.crypto_utils import CryptoUtils

class TestCryptoUtils(unittest.TestCase):
    """Test suite for cryptographic utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.crypto_utils = CryptoUtils()
        
    def test_entropy_calculation(self):
        """Test entropy calculation"""
        # Test uniform distribution
        uniform_data = bytes([i % 256 for i in range(1000)])
        entropy = self.crypto_utils.calculate_entropy(uniform_data)
        self.assertAlmostEqual(entropy, 8.0, places=1)
        
        # Test zero entropy
        zero_data = bytes([0] * 1000)
        entropy = self.crypto_utils.calculate_entropy(zero_data)
        self.assertEqual(entropy, 0.0)
        
        # Test random data
        random_data = np.random.bytes(1000)
        entropy = self.crypto_utils.calculate_entropy(random_data)
        self.assertGreater(entropy, 7.0)
    
    def test_key_derivation(self):
        """Test key derivation methods"""
        password = b"test_password"
        salt = b"test_salt"
        
        # Test different versions
        key_03x = self.crypto_utils.derive_key(password, salt, "0.3.0")
        key_04x = self.crypto_utils.derive_key(password, salt, "0.4.0")
        key_modern = self.crypto_utils.derive_key(password, salt, "0.10.0")
        
        # Verify different results
        self.assertNotEqual(key_03x, key_04x)
        self.assertNotEqual(key_04x, key_modern)
        self.assertEqual(len(key_modern), 32)
    
    def test_aes_decryption(self):
        """Test AES decryption"""
        key = b"0" * 32
        iv = b"0" * 16
        plaintext = b"test message"
        
        # Test different padding modes
        for padding_mode in ['pkcs7', 'zero']:
            # Encrypt
            cipher = self.crypto_utils.encrypt_aes(key, iv, plaintext, padding_mode)
            # Decrypt
            decrypted = self.crypto_utils.decrypt_aes(key, iv, cipher, padding_mode)
            self.assertEqual(decrypted, plaintext)
    
    def test_key_pair_verification(self):
        """Test key pair verification"""
        from ecdsa import SigningKey, SECP256k1
        
        # Generate test key pair
        private_key = SigningKey.generate(curve=SECP256k1)
        public_key = private_key.get_verifying_key()
        
        # Test valid pair
        self.assertTrue(self.crypto_utils.verify_key_pair(
            private_key.to_string(),
            public_key.to_string()
        ))
        
        # Test invalid pair
        invalid_private = SigningKey.generate(curve=SECP256k1).to_string()
        self.assertFalse(self.crypto_utils.verify_key_pair(
            invalid_private,
            public_key.to_string()
        ))
    
    def test_pattern_analysis(self):
        """Test encryption pattern analysis"""
        # Test Bitcoin Core pattern
        btc_pattern = b"\x01\x42" + b"0" * 30
        analysis = self.crypto_utils.analyze_key_pattern(btc_pattern)
        self.assertIn(('bitcoin_core_early', 0.9), analysis['potential_patterns'])
        
        # Test random data
        random_data = np.random.bytes(32)
        analysis = self.crypto_utils.analyze_key_pattern(random_data)
        self.assertGreater(analysis['entropy'], 7.0)
        
    def test_error_handling(self):
        """Test error handling"""
        # Test invalid inputs
        self.assertIsNone(self.crypto_utils.decrypt_aes(b"", b"", b""))
        self.assertFalse(self.crypto_utils.verify_key_pair(b"", b""))
        
        # Test empty data
        self.assertEqual(self.crypto_utils.calculate_entropy(b""), 0.0)
        self.assertEqual(len(self.crypto_utils.analyze_key_pattern(b"")['potential_patterns']), 0)