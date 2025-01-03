import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..utils.crypto_utils import CryptoUtils
from ..utils.logging import setup_logger
import json
from pathlib import Path
from scipy import stats
from collections import Counter

class EncryptionAnalyzer:
    """Analyzer for encryption characteristics"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.signatures = self._load_encryption_signatures()
        
    def _analyze_block_characteristics(self, data: bytes) -> Dict[str, any]:
        """Analyze block-level characteristics of encrypted data
        
        Args:
            data: Encrypted data bytes
            
        Returns:
            Dictionary containing block analysis results
        """
        try:
            block_analysis = {
                'block_sizes': [],
                'block_entropy': [],
                'block_patterns': {},
                'block_correlations': [],
                'anomalies': []
            }
            
            # Detect block size
            potential_sizes = self._detect_block_sizes(data)
            block_analysis['block_sizes'] = potential_sizes
            
            # Analyze blocks
            most_likely_size = potential_sizes[0] if potential_sizes else 16
            blocks = [data[i:i+most_likely_size] 
                     for i in range(0, len(data), most_likely_size)]
            
            # Calculate block entropy
            for block in blocks:
                entropy = self._calculate_block_entropy(block)
                block_analysis['block_entropy'].append(entropy)
                
            # Detect patterns
            block_analysis['block_patterns'] = self._detect_block_patterns(blocks)
            
            # Calculate correlations
            block_analysis['block_correlations'] = self._calculate_block_correlations(blocks)
            
            # Detect anomalies
            block_analysis['anomalies'] = self._detect_block_anomalies(blocks)
            
            return block_analysis
            
        except Exception as e:
            self.logger.error(f"Error in block analysis: {str(e)}")
            return {}

    def _analyze_padding(self, data: bytes) -> Dict[str, any]:
        """Analyze padding characteristics of encrypted data
        
        Args:
            data: Encrypted data bytes
            
        Returns:
            Dictionary containing padding analysis results
        """
        try:
            padding_analysis = {
                'padding_type': None,
                'padding_length': 0,
                'padding_pattern': None,
                'is_valid': False,
                'confidence': 0.0
            }
            
            # Check PKCS7 padding
            pkcs7_result = self._check_pkcs7_padding(data)
            if pkcs7_result['is_valid']:
                padding_analysis.update(pkcs7_result)
                padding_analysis['padding_type'] = 'PKCS7'
                return padding_analysis
                
            # Check ISO/IEC 7816-4 padding
            iso7816_result = self._check_iso7816_padding(data)
            if iso7816_result['is_valid']:
                padding_analysis.update(iso7816_result)
                padding_analysis['padding_type'] = 'ISO7816-4'
                return padding_analysis
                
            # Check zero padding
            zero_result = self._check_zero_padding(data)
            if zero_result['is_valid']:
                padding_analysis.update(zero_result)
                padding_analysis['padding_type'] = 'ZERO'
                return padding_analysis
                
            # Check for custom padding
            custom_result = self._analyze_custom_padding(data)
            if custom_result['is_valid']:
                padding_analysis.update(custom_result)
                padding_analysis['padding_type'] = 'CUSTOM'
                
            return padding_analysis
            
        except Exception as e:
            self.logger.error(f"Error in padding analysis: {str(e)}")
            return {}

    def _run_statistical_tests(self, data: bytes) -> Dict[str, float]:
        """Run statistical tests on encrypted data
        
        Args:
            data: Encrypted data bytes
            
        Returns:
            Dictionary containing test results
        """
        try:
            results = {}
            
            # Convert data to numpy array
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            # Frequency test
            results['frequency_test'] = self._run_frequency_test(data_array)
            
            # Runs test
            results['runs_test'] = self._run_runs_test(data_array)
            
            # Serial test
            results['serial_test'] = self._run_serial_test(data_array)
            
            # Entropy test
            results['entropy_test'] = self._calculate_entropy_score(data_array)
            
            # Chi-square test
            results['chi_square_test'] = self._run_chi_square_test(data_array)
            
            # Autocorrelation test
            results['autocorrelation'] = self._run_autocorrelation_test(data_array)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in statistical tests: {str(e)}")
            return {}

    def _load_encryption_signatures(self) -> Dict[str, Dict[str, any]]:
        """Load known encryption method signatures
        
        Returns:
            Dictionary containing encryption signatures
        """
        try:
            signatures = {
                'AES': {
                    'block_sizes': [16, 24, 32],
                    'patterns': {
                        'header': [b'\x00\x01', b'\x00\x02'],
                        'padding': ['PKCS7', 'ISO7816-4'],
                        'mode_indicators': {
                            'CBC': {'iv_present': True, 'block_correlation': 'high'},
                            'ECB': {'iv_present': False, 'block_correlation': 'low'},
                            'CFB': {'iv_present': True, 'stream_like': True}
                        }
                    },
                    'statistical_properties': {
                        'entropy_range': (7.8, 8.0),
                        'chi_square_threshold': 0.01
                    }
                },
                'DES': {
                    'block_sizes': [8],
                    'patterns': {
                        'header': [b'\x00\x00'],
                        'padding': ['PKCS7'],
                        'mode_indicators': {
                            'CBC': {'iv_present': True},
                            'ECB': {'iv_present': False}
                        }
                    },
                    'statistical_properties': {
                        'entropy_range': (7.6, 7.9),
                        'chi_square_threshold': 0.01
                    }
                },
                'Blowfish': {
                    'block_sizes': [8],
                    'patterns': {
                        'header': [b'\x00\x03'],
                        'padding': ['PKCS7', 'ZERO'],
                        'mode_indicators': {
                            'CBC': {'iv_present': True},
                            'ECB': {'iv_present': False}
                        }
                    },
                    'statistical_properties': {
                        'entropy_range': (7.7, 7.95),
                        'chi_square_threshold': 0.01
                    }
                }
            }
            
            # Load custom signatures if available
            custom_signatures_path = Path(self.config.get('signatures_path', 'signatures.json'))
            if custom_signatures_path.exists():
                with open(custom_signatures_path, 'r') as f:
                    custom_signatures = json.load(f)
                signatures.update(custom_signatures)
                
            return signatures
            
        except Exception as e:
            self.logger.error(f"Error loading encryption signatures: {str(e)}")
            return {}

    def _detect_block_sizes(self, data: bytes) -> List[int]:
        """Detect potential block sizes"""
        potential_sizes = []
        for size in [8, 16, 24, 32]:
            if len(data) % size == 0:
                score = self._evaluate_block_size(data, size)
                potential_sizes.append((size, score))
        return [size for size, _ in sorted(potential_sizes, key=lambda x: x[1], reverse=True)]

    def _calculate_block_entropy(self, block: bytes) -> float:
        """Calculate entropy for a single block"""
        counter = Counter(block)
        entropy = 0
        for count in counter.values():
            p = count / len(block)
            entropy -= p * np.log2(p)
        return entropy

    def _detect_block_patterns(self, blocks: List[bytes]) -> Dict[str, any]:
        """Detect patterns in block sequence"""
        patterns = {
            'repeating_blocks': self._find_repeating_blocks(blocks),
            'sequential_patterns': self._find_sequential_patterns(blocks),
            'block_similarity': self._calculate_block_similarity(blocks)
        }
        return patterns

    def _calculate_block_correlations(self, blocks: List[bytes]) -> List[float]:
        """Calculate correlations between consecutive blocks"""
        correlations = []
        for i in range(len(blocks)-1):
            correlation = np.corrcoef(
                np.frombuffer(blocks[i], dtype=np.uint8),
                np.frombuffer(blocks[i+1], dtype=np.uint8)
            )[0,1]
            correlations.append(correlation)
        return correlations