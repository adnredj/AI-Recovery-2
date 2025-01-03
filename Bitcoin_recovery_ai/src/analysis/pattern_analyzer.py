import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..utils.crypto_utils import CryptoUtils
from ..utils.logging import setup_logger
from collections import OrderedDict

class PatternAnalyzer:
    """Advanced pattern analysis for wallet recovery"""
    
    def __init__(self, config):
        self.config = config
        self.crypto_utils = CryptoUtils()
        self.logger = setup_logger(__name__)
        self.patterns = self._load_pattern_definitions()
        
    def analyze_key_patterns(self, key_data: bytes) -> Dict[str, any]:
        """Analyze encryption key patterns"""
        analysis = {
            'byte_patterns': self._analyze_byte_patterns(key_data),
            'structure_patterns': self._analyze_structure_patterns(key_data),
            'entropy_patterns': self._analyze_entropy_patterns(key_data),
            'version_indicators': self._detect_version_indicators(key_data),
            'potential_methods': []
        }
        
        # Combine analyses to identify potential encryption methods
        analysis['potential_methods'] = self._identify_encryption_methods(analysis)
        return analysis
    
    def _analyze_byte_patterns(self, data: bytes) -> Dict[str, any]:
        """Analyze byte-level patterns"""
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        return {
            'repeating_sequences': self._find_repeating_sequences(byte_array),
            'byte_frequency': self._analyze_byte_frequency(byte_array),
            'special_bytes': self._identify_special_bytes(byte_array),
            'sequence_patterns': self._analyze_sequence_patterns(byte_array)
        }
    
    def _analyze_structure_patterns(self, data: bytes) -> Dict[str, any]:
        """Analyze structural patterns"""
        return {
            'header_pattern': self._analyze_header(data[:8]),
            'block_structure': self._analyze_block_structure(data),
            'padding_pattern': self._analyze_padding(data),
            'key_format': self._identify_key_format(data)
        }
    
    def _analyze_entropy_patterns(self, data: bytes) -> Dict[str, List[float]]:
        """Analyze entropy patterns in different parts of the data"""
        window_size = 16
        entropy_patterns = []
        
        # Sliding window entropy analysis
        for i in range(0, len(data) - window_size + 1):
            window = data[i:i + window_size]
            entropy = self.crypto_utils.calculate_entropy(window)
            entropy_patterns.append(entropy)
            
        return {
            'entropy_profile': entropy_patterns,
            'local_variations': self._analyze_entropy_variations(entropy_patterns),
            'entropy_anomalies': self._detect_entropy_anomalies(entropy_patterns)
        }
    
    def _find_repeating_sequences(self, data: np.ndarray) -> List[Dict[str, any]]:
        """Find repeating byte sequences"""
        sequences = []
        min_length = 3
        max_length = 16
        
        for length in range(min_length, max_length + 1):
            for i in range(len(data) - length + 1):
                sequence = data[i:i + length]
                count = 0
                positions = []
                
                for j in range(len(data) - length + 1):
                    if np.array_equal(data[j:j + length], sequence):
                        count += 1
                        positions.append(j)
                
                if count > 1:  # Only record if sequence repeats
                    sequences.append({
                        'sequence': sequence.tobytes(),
                        'length': length,
                        'count': count,
                        'positions': positions
                    })
        
        return sequences
    
    def _analyze_byte_frequency(self, data: np.ndarray) -> Dict[int, float]:
        """Analyze byte frequency distribution"""
        counts = np.bincount(data, minlength=256)
        frequencies = counts / len(data)
        
        return {
            'distribution': {i: float(freq) for i, freq in enumerate(frequencies) if freq > 0},
            'most_common': np.argsort(counts)[-5:].tolist(),
            'least_common': np.argsort(counts)[:5].tolist(),
            'zero_bytes': np.where(counts == 0)[0].tolist()
        }
    
    def _identify_special_bytes(self, data: np.ndarray) -> Dict[str, List[int]]:
        """Identify special byte patterns"""
        return {
            'null_bytes': np.where(data == 0)[0].tolist(),
            'marker_bytes': np.where(data == 0x42)[0].tolist(),  # Bitcoin Core marker
            'version_bytes': np.where(data == 0x01)[0].tolist(),  # Version indicator
            'padding_bytes': np.where(data == 0x80)[0].tolist()  # Common padding byte
        }
    
    def _analyze_sequence_patterns(self, data: np.ndarray) -> Dict[str, any]:
        """Analyze sequential patterns in the data"""
        differences = np.diff(data)
        
        return {
            'increasing_sequences': self._find_monotonic_sequences(differences, 1),
            'decreasing_sequences': self._find_monotonic_sequences(differences, -1),
            'constant_sequences': self._find_constant_sequences(data),
            'pattern_lengths': self._analyze_pattern_lengths(differences)
        }
    
    def _find_monotonic_sequences(self, differences: np.ndarray, direction: int) -> List[Dict[str, any]]:
        """Find monotonic sequences in the data"""
        sequences = []
        start = 0
        
        for i in range(1, len(differences)):
            if differences[i] * direction < 0:
                if i - start > 2:  # Minimum sequence length
                    sequences.append({
                        'start': start,
                        'length': i - start,
                        'slope': float(np.mean(differences[start:i]))
                    })
                start = i
                
        return sequences
    
    def _identify_encryption_methods(self, analysis: Dict[str, any]) -> List[Dict[str, float]]:
        """Identify potential encryption methods based on patterns"""
        potential_methods = []
        
        # Check for Bitcoin Core patterns
        if self._matches_bitcoin_core_pattern(analysis):
            potential_methods.append({
                'method': 'bitcoin_core',
                'confidence': self._calculate_pattern_confidence(analysis, 'bitcoin_core'),
                'version': self._estimate_version(analysis)
            })
            
        # Check for AES patterns
        if self._matches_aes_pattern(analysis):
            potential_methods.append({
                'method': 'aes',
                'confidence': self._calculate_pattern_confidence(analysis, 'aes'),
                'mode': self._identify_aes_mode(analysis)
            })
            
        # Add more encryption method patterns as needed
        
        return sorted(potential_methods, key=lambda x: x['confidence'], reverse=True)
    
    def _matches_bitcoin_core_pattern(self, analysis: Dict[str, any]) -> bool:
        """Check if patterns match Bitcoin Core encryption
        
        Args:
            analysis: Dictionary containing pattern analysis
            
        Returns:
            Boolean indicating if patterns match Bitcoin Core
        """
        try:
            # Check header pattern
            if 'header_bytes' in analysis:
                header = analysis['header_bytes']
                if not header.startswith(b'\x01\x42\x49\x54'):  # "BIT" magic bytes
                    return False
                    
            # Check key structure
            if 'key_structure' in analysis:
                structure = analysis['key_structure']
                # Check for characteristic Bitcoin Core key structure
                if not (structure.get('has_salt', False) and
                       structure.get('has_checksum', False)):
                    return False
                    
            # Check encryption markers
            markers = analysis.get('encryption_markers', {})
            required_markers = {
                'key_length': 32,
                'salt_present': True,
                'version_byte': True
            }
            
            return all(
                markers.get(key) == value
                for key, value in required_markers.items()
            )
            
        except Exception as e:
            self.logger.error(f"Error in Bitcoin Core pattern matching: {str(e)}")
            return False

    def _matches_aes_pattern(self, analysis: Dict[str, any]) -> bool:
        """Check if patterns match AES encryption
        
        Args:
            analysis: Dictionary containing pattern analysis
            
        Returns:
            Boolean indicating if patterns match AES
        """
        try:
            # Check block size
            if 'block_analysis' in analysis:
                block_size = analysis['block_analysis'].get('size', 0)
                if block_size not in [16, 24, 32]:  # AES block sizes
                    return False
                    
            # Check for IV presence
            if 'iv_analysis' in analysis:
                iv_data = analysis['iv_analysis']
                if not (iv_data.get('present', False) and
                       iv_data.get('size', 0) == 16):  # AES IV size
                    return False
                    
            # Check padding
            if 'padding_analysis' in analysis:
                padding = analysis['padding_analysis']
                if not padding.get('pkcs7_compatible', False):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error in AES pattern matching: {str(e)}")
            return False

    def _calculate_pattern_confidence(self, analysis: Dict[str, any], method: str) -> float:
        """Calculate confidence score for pattern match
        
        Args:
            analysis: Dictionary containing pattern analysis
            method: Encryption method to calculate confidence for
            
        Returns:
            Float confidence score between 0 and 1
        """
        try:
            confidence_factors = []
            
            if method == 'bitcoin_core':
                # Bitcoin Core specific factors
                factors = {
                    'header_match': 0.3,
                    'structure_match': 0.3,
                    'marker_match': 0.2,
                    'version_match': 0.2
                }
                
                # Calculate each factor
                if 'header_bytes' in analysis:
                    confidence_factors.append(
                        factors['header_match'] * 
                        self._calculate_header_confidence(analysis)
                    )
                    
                if 'key_structure' in analysis:
                    confidence_factors.append(
                        factors['structure_match'] * 
                        self._calculate_structure_confidence(analysis)
                    )
                    
                if 'encryption_markers' in analysis:
                    confidence_factors.append(
                        factors['marker_match'] * 
                        self._calculate_marker_confidence(analysis)
                    )
                    
                if 'version_indicators' in analysis:
                    confidence_factors.append(
                        factors['version_match'] * 
                        self._calculate_version_confidence(analysis)
                    )
                    
            elif method == 'aes':
                # AES specific factors
                factors = {
                    'block_size': 0.3,
                    'iv_analysis': 0.3,
                    'padding': 0.2,
                    'entropy': 0.2
                }
                
                # Calculate each factor
                if 'block_analysis' in analysis:
                    confidence_factors.append(
                        factors['block_size'] * 
                        self._calculate_block_confidence(analysis)
                    )
                    
                if 'iv_analysis' in analysis:
                    confidence_factors.append(
                        factors['iv_analysis'] * 
                        self._calculate_iv_confidence(analysis)
                    )
                    
                if 'padding_analysis' in analysis:
                    confidence_factors.append(
                        factors['padding'] * 
                        self._calculate_padding_confidence(analysis)
                    )
                    
                if 'entropy_analysis' in analysis:
                    confidence_factors.append(
                        factors['entropy'] * 
                        self._calculate_entropy_confidence(analysis)
                    )
                    
            return sum(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {str(e)}")
            return 0.0

    def _estimate_version(self, analysis: Dict[str, any]) -> str:
        """Estimate Bitcoin Core version from patterns
        
        Args:
            analysis: Dictionary containing pattern analysis
            
        Returns:
            String indicating estimated version
        """
        try:
            version_indicators = analysis.get('version_indicators', {})
            
            # Check explicit version byte
            if 'version_byte' in version_indicators:
                version_map = {
                    0x01: '0.4.0',
                    0x02: '0.5.0',
                    0x03: '0.6.0'
                }
                version_byte = version_indicators['version_byte']
                if version_byte in version_map:
                    return version_map[version_byte]
                    
            # Check encryption method
            if 'encryption_method' in version_indicators:
                method = version_indicators['encryption_method']
                if 'sha256' in method.lower():
                    return 'pre-0.4.0'
                elif 'aes' in method.lower():
                    return 'post-0.4.0'
                    
            # Check structure version
            if 'structure_version' in version_indicators:
                structure = version_indicators['structure_version']
                if 'berkeley' in structure.lower():
                    return 'pre-0.8.0'
                elif 'sqlite' in structure.lower():
                    return 'post-0.8.0'
                    
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Error estimating version: {str(e)}")
            return 'error'

    def _identify_aes_mode(self, analysis: Dict[str, any]) -> str:
        """Identify AES mode of operation
        
        Args:
            analysis: Dictionary containing pattern analysis
            
        Returns:
            String indicating AES mode
        """
        try:
            # Check IV presence
            if analysis.get('iv_analysis', {}).get('present', False):
                # Check for characteristic CBC patterns
                if self._check_cbc_patterns(analysis):
                    return 'CBC'
                    
                # Check for characteristic CFB patterns
                if self._check_cfb_patterns(analysis):
                    return 'CFB'
                    
            else:
                # Check for characteristic ECB patterns
                if self._check_ecb_patterns(analysis):
                    return 'ECB'
                    
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Error identifying AES mode: {str(e)}")
            return 'error'

    def _check_cbc_patterns(self, analysis: Dict[str, any]) -> bool:
        """Check for CBC mode patterns"""
        block_patterns = analysis.get('block_patterns', {})
        return (block_patterns.get('block_dependency', False) and
                block_patterns.get('iv_dependency', False))

    def _check_cfb_patterns(self, analysis: Dict[str, any]) -> bool:
        """Check for CFB mode patterns"""
        return analysis.get('stream_characteristics', {}).get('cfb_pattern', False)

    def _check_ecb_patterns(self, analysis: Dict[str, any]) -> bool:
        """Check for ECB mode patterns"""
        return analysis.get('block_patterns', {}).get('independent_blocks', False)