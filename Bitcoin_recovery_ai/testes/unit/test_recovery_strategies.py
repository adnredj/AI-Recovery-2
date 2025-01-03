import unittest
import torch
from pathlib import Path
from typing import Dict, List, Optional
import json
import hashlib
from Crypto.Cipher import AES
from ..src.recovery.strategies import RecoveryStrategies
from ..src.utils.crypto_utils import CryptoUtils

class TestRecoveryStrategies(unittest.TestCase):
    """Test suite for wallet recovery strategies"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = self._load_test_config()
        self.strategies = RecoveryStrategies(self.config)
        self.crypto_utils = CryptoUtils()
        self.test_data_dir = Path('tests/test_data')
        self.test_data_dir.mkdir(exist_ok=True)
        
    def test_bitcoin_core_2010_recovery(self):
        """Test Bitcoin Core 2010 recovery strategy"""
        # Load test wallet data
        wallet_data = self._load_test_wallet('bitcoin_core_2010.dat')
        predictions = self._generate_test_predictions()
        
        # Execute recovery
        result = self.strategies._recover_bitcoin_core_2010(wallet_data, predictions)
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertGreater(len(result['potential_keys']), 0)
        self.assertGreater(result['confidence'], self.config.thresholds.confidence_threshold)
        
        # Verify key validity
        for key in result['potential_keys']:
            self.assertTrue(self.crypto_utils.verify_key_pair(
                key['private_key'],
                wallet_data['master_key']['public_key']
            ))
    
    def test_bitcoin_core_2011_recovery(self):
        """Test Bitcoin Core 2011 recovery strategy"""
        wallet_data = self._load_test_wallet('bitcoin_core_2011.dat')
        predictions = self._generate_test_predictions()
        
        result = self.strategies._recover_bitcoin_core_2011(wallet_data, predictions)
        
        self.assertTrue(result['success'])
        self.assertGreater(len(result['potential_keys']), 0)
        self.assertTrue(all(
            self._verify_key_derivation(key['derived_key'], wallet_data)
            for key in result['potential_keys']
        ))
    
    def test_berkeley_db_recovery(self):
        """Test Berkeley DB recovery strategy"""
        wallet_data = self._load_test_wallet('berkeley_db.dat')
        predictions = self._generate_test_predictions()
        
        result = self.strategies._recover_berkeley_db(wallet_data, predictions)
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['db_analysis'])
        self.assertGreater(len(result['recovered_keys']), 0)
    
    def test_encrypted_key_recovery(self):
        """Test encrypted key recovery strategy"""
        wallet_data = self._load_test_wallet('encrypted.dat')
        predictions = self._generate_test_predictions()
        
        result = self.strategies._recover_encrypted_key(wallet_data, predictions)
        
        self.assertTrue(result['success'])
        self.assertGreater(len(result['attempts']), 0)
        self.assertIsNotNone(result['encryption_analysis'])
    
    def test_invalid_wallet_handling(self):
        """Test handling of invalid wallet data"""
        invalid_data = {'master_key': None}
        predictions = self._generate_test_predictions()
        
        for strategy in ['bitcoin_core_2010', 'bitcoin_core_2011', 'berkeley_db', 'encrypted_key']:
            result = self.strategies.execute_strategy(strategy, invalid_data, predictions)
            self.assertFalse(result['success'])
            self.assertIn('error', result)
    
    def _load_test_config(self) -> Dict:
        """Load test configuration"""
        return {
            'thresholds': {
                'confidence_threshold': 0.8,
                'pattern_match_threshold': 0.9
            }
        }
    
    def _load_test_wallet(self, filename: str) -> Dict:
        """Load test wallet data from file
        
        Args:
            filename: Name of test wallet file
            
        Returns:
            Dictionary containing wallet data
        """
        try:
            file_path = self.test_data_dir / filename
            
            if not file_path.exists():
                return self._generate_mock_wallet_data(filename)
                
            if file_path.suffix == '.dat':
                with open(file_path, 'rb') as f:
                    data = f.read()
                    return self._parse_wallet_data(data)
                    
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
                    
            else:
                self.fail(f"Unsupported test wallet format: {file_path.suffix}")
                
        except Exception as e:
            self.fail(f"Error loading test wallet {filename}: {str(e)}")

    def _generate_test_predictions(self) -> Dict[str, torch.Tensor]:
        """Generate test prediction data
        
        Returns:
            Dictionary containing test predictions
        """
        base_predictions = {
            'recovery_probability': torch.tensor(0.95),
            'encryption_pattern': torch.randn(1, 10),
            'pattern_confidence': torch.tensor(0.9),
            'strategy_scores': {
                'bitcoin_core_2010': torch.tensor(0.8),
                'bitcoin_core_2011': torch.tensor(0.7),
                'berkeley_db': torch.tensor(0.6),
                'encrypted_key': torch.tensor(0.5)
            }
        }
        
        # Add additional test predictions
        base_predictions.update({
            'version_confidence': torch.tensor(0.85),
            'pattern_matches': torch.tensor([0.9, 0.8, 0.7, 0.6]),
            'encryption_strength': torch.tensor(0.75)
        })
        
        return base_predictions

    def _verify_key_derivation(self, derived_key: bytes, wallet_data: Dict) -> bool:
        """Verify if derived key is valid for the wallet
        
        Args:
            derived_key: Derived key bytes
            wallet_data: Wallet data dictionary
            
        Returns:
            Boolean indicating if key is valid
        """
        try:
            # Verify key format
            if not self._verify_key_format(derived_key):
                return False
                
            # Check against known test vectors
            if 'test_vectors' in wallet_data:
                return self._verify_against_test_vectors(
                    derived_key,
                    wallet_data['test_vectors']
                )
                
            # Verify encryption if available
            if 'encrypted_data' in wallet_data:
                return self._verify_decryption(
                    derived_key,
                    wallet_data['encrypted_data'],
                    wallet_data.get('iv')
                )
                
            # Verify key hash if available
            if 'key_hash' in wallet_data:
                derived_hash = hashlib.sha256(derived_key).hexdigest()
                return derived_hash == wallet_data['key_hash']
                
            return True
            
        except Exception as e:
            self.fail(f"Error verifying key derivation: {str(e)}")

    def _generate_mock_wallet_data(self, wallet_type: str) -> Dict:
        """Generate mock wallet data for testing
        
        Args:
            wallet_type: Type of wallet to mock
            
        Returns:
            Dictionary containing mock wallet data
        """
        if 'bitcoin_core_2010' in wallet_type:
            return {
                'master_key': {
                    'encrypted_key': b'test_encrypted_key',
                    'public_key': b'test_public_key',
                    'salt': b'test_salt'
                },
                'test_vectors': [
                    {
                        'input': b'test_input',
                        'expected': b'test_output'
                    }
                ]
            }
            
        elif 'bitcoin_core_2011' in wallet_type:
            return {
                'master_key': {
                    'encrypted_key': b'test_encrypted_key',
                    'salt': b'test_salt',
                    'iv': b'test_iv'
                },
                'encrypted_data': b'test_encrypted_data'
            }
            
        elif 'berkeley_db' in wallet_type:
            return {
                'path': 'test_path',
                'keys': [
                    {'key_data': b'test_key_1'},
                    {'key_data': b'test_key_2'}
                ]
            }
            
        elif 'encrypted' in wallet_type:
            return {
                'master_key': {
                    'encrypted_key': b'test_encrypted_key',
                    'method': 'aes-256-cbc',
                    'key_hash': 'test_hash'
                }
            }
            
        else:
            self.fail(f"Unknown wallet type for mocking: {wallet_type}")

    def _verify_key_format(self, key: bytes) -> bool:
        """Verify if key has valid format"""
        return (
            isinstance(key, bytes) and
            len(key) == 32 and
            all(0 <= b <= 255 for b in key)
        )

    def _verify_against_test_vectors(self,
                                   derived_key: bytes,
                                   test_vectors: List[Dict]) -> bool:
        """Verify key against test vectors"""
        for vector in test_vectors:
            if not self._verify_test_vector(derived_key, vector):
                return False
        return True

    def _verify_decryption(self,
                          key: bytes,
                          encrypted_data: bytes,
                          iv: Optional[bytes]) -> bool:
        """Verify key by attempting decryption"""
        try:
            if iv:
                cipher = AES.new(key, AES.MODE_CBC, iv)
            else:
                cipher = AES.new(key, AES.MODE_ECB)
                
            decrypted = cipher.decrypt(encrypted_data)
            return self._verify_decrypted_data(decrypted)
            
        except Exception:
            return False

    def _verify_decrypted_data(self, data: bytes) -> bool:
        """Verify if decrypted data is valid"""
        return (
            len(data) >= 32 and
            data.startswith(b'\x01') and  # Version byte
            data[-4:] == hashlib.sha256(data[:-4]).digest()[:4]  # Checksum
        )