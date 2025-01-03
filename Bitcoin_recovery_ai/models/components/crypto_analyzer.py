import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import numpy as np
from scipy.stats import entropy
from collections import Counter
import hashlib
from typing import Dict, List, Optional, Tuple
import struct

class CryptoAnalyzer(nn.Module):
    """Cryptographic analysis component"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Entropy analysis network
        self.entropy_analyzer = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Key pattern detector
        self.key_pattern_detector = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def analyze_encryption(self, data, iv=None):
        """Analyze encryption patterns in data"""
        entropy = self._calculate_entropy(data)
        key_patterns = self._detect_key_patterns(data)
        
        return {
            'entropy': entropy,
            'key_patterns': key_patterns,
            'possible_methods': self._identify_encryption_methods(data, iv)
        }

    def _calculate_entropy(self, data: bytes) -> Dict[str, float]:
        """Calculate Shannon entropy and related metrics of data
        
        Args:
            data: Bytes of data to analyze
            
        Returns:
            Dictionary containing entropy metrics
        """
        try:
            # Convert to numpy array for calculations
            byte_array = np.frombuffer(data, dtype=np.uint8)
            
            # Calculate byte frequency distribution
            counts = Counter(byte_array)
            frequencies = np.array(list(counts.values())) / len(byte_array)
            
            # Calculate Shannon entropy
            shannon_entropy = entropy(frequencies, base=2)
            
            # Calculate block entropy (for 16-byte blocks)
            block_size = 16
            blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
            block_counts = Counter(blocks)
            block_frequencies = np.array(list(block_counts.values())) / len(blocks)
            block_entropy = entropy(block_frequencies, base=2)
            
            # Calculate entropy rate
            entropy_rate = shannon_entropy / 8  # bits per byte
            
            # Calculate compression ratio estimate
            theoretical_size = len(data) * shannon_entropy / 8
            compression_ratio = len(data) / theoretical_size
            
            return {
                'shannon_entropy': shannon_entropy,
                'block_entropy': block_entropy,
                'entropy_rate': entropy_rate,
                'compression_ratio': compression_ratio,
                'byte_distribution': dict(counts),
                'randomness_score': min(shannon_entropy / 8, 1.0)  # Normalized 0-1
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {str(e)}")
            raise

    def _detect_key_patterns(self, data: bytes) -> Dict[str, any]:
        """Detect potential key derivation patterns
        
        Args:
            data: Bytes of key data to analyze
            
        Returns:
            Dictionary containing detected patterns
        """
        try:
            patterns = {
                'repeated_sequences': [],
                'byte_patterns': [],
                'structure_hints': [],
                'derivation_method': None,
                'confidence': 0.0
            }
            
            # Check for repeated sequences
            for length in range(4, 33):  # Check sequences up to 32 bytes
                sequences = self._find_repeated_sequences(data, length)
                if sequences:
                    patterns['repeated_sequences'].extend(sequences)
            
            # Analyze byte patterns
            byte_patterns = self._analyze_byte_patterns(data)
            patterns['byte_patterns'] = byte_patterns
            
            # Check for known key derivation methods
            derivation_info = self._check_derivation_methods(data)
            patterns.update(derivation_info)
            
            # Check for structural patterns
            structure_patterns = self._analyze_structure(data)
            patterns['structure_hints'] = structure_patterns
            
            # Calculate confidence score
            patterns['confidence'] = self._calculate_pattern_confidence(patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting key patterns: {str(e)}")
            raise

    def _identify_encryption_methods(self, data: bytes, iv: Optional[bytes] = None) -> Dict[str, any]:
        """Identify possible encryption methods used
        
        Args:
            data: Encrypted data bytes
            iv: Optional initialization vector
            
        Returns:
            Dictionary containing identified encryption methods and confidence scores
        """
        try:
            results = {
                'likely_methods': [],
                'iv_analysis': {},
                'block_analysis': {},
                'confidence_scores': {}
            }
            
            # Analyze IV if provided
            if iv:
                iv_analysis = self._analyze_iv(iv)
                results['iv_analysis'] = iv_analysis
            
            # Analyze block characteristics
            block_analysis = self._analyze_blocks(data)
            results['block_analysis'] = block_analysis
            
            # Check for known encryption signatures
            encryption_signatures = self._check_encryption_signatures(data)
            
            # Identify likely encryption methods
            likely_methods = []
            confidence_scores = {}
            
            # AES detection
            aes_score = self._check_aes_characteristics(data, iv)
            if aes_score > 0.6:
                likely_methods.append('AES')
                confidence_scores['AES'] = aes_score
            
            # Blowfish detection
            blowfish_score = self._check_blowfish_characteristics(data)
            if blowfish_score > 0.6:
                likely_methods.append('Blowfish')
                confidence_scores['Blowfish'] = blowfish_score
            
            # ChaCha20 detection
            chacha_score = self._check_chacha_characteristics(data)
            if chacha_score > 0.6:
                likely_methods.append('ChaCha20')
                confidence_scores['ChaCha20'] = chacha_score
            
            results['likely_methods'] = likely_methods
            results['confidence_scores'] = confidence_scores
            
            # Add encryption mode hints
            results['mode_hints'] = self._detect_encryption_mode(data, iv)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error identifying encryption methods: {str(e)}")
            raise

    def _find_repeated_sequences(self, data: bytes, length: int) -> List[Dict[str, any]]:
        """Find repeated sequences of specified length"""
        sequences = []
        seen = {}
        
        for i in range(len(data) - length + 1):
            seq = data[i:i+length]
            if seq in seen:
                sequences.append({
                    'sequence': seq.hex(),
                    'length': length,
                    'positions': [seen[seq], i]
                })
            else:
                seen[seq] = i
                
        return sequences

    def _analyze_byte_patterns(self, data: bytes) -> Dict[str, any]:
        """Analyze byte-level patterns"""
        patterns = {
            'byte_frequency': Counter(data),
            'byte_pairs': Counter(zip(data[:-1], data[1:])),
            'alignment_patterns': [],
            'special_bytes': []
        }
        
        # Check byte alignment patterns
        for offset in range(8):
            aligned_bytes = data[offset::8]
            if len(aligned_bytes) > 16:
                patterns['alignment_patterns'].append({
                    'offset': offset,
                    'entropy': self._calculate_entropy(aligned_bytes)['shannon_entropy']
                })
        
        return patterns

    def _check_derivation_methods(self, data: bytes) -> Dict[str, any]:
        """Check for known key derivation methods"""
        results = {
            'derivation_method': None,
            'method_confidence': 0.0,
            'parameters': {}
        }
        
        # Check PBKDF2 characteristics
        if self._check_pbkdf2_pattern(data):
            results['derivation_method'] = 'PBKDF2'
            results['method_confidence'] = 0.85
            
        # Check scrypt characteristics
        elif self._check_scrypt_pattern(data):
            results['derivation_method'] = 'scrypt'
            results['method_confidence'] = 0.80
            
        return results

    def _analyze_structure(self, data: bytes) -> List[Dict[str, any]]:
        """Analyze data structure patterns"""
        structures = []
        
        # Check for header patterns
        if len(data) >= 16:
            header = data[:16]
            header_analysis = {
                'type': 'header',
                'pattern': header.hex(),
                'entropy': self._calculate_entropy(header)['shannon_entropy']
            }
            structures.append(header_analysis)
        
        # Check for block boundaries
        for block_size in [16, 32, 64]:
            if len(data) % block_size == 0:
                structures.append({
                    'type': 'block_alignment',
                    'size': block_size
                })
                
        return structures

    def _calculate_pattern_confidence(self, patterns: Dict[str, any]) -> float:
        """Calculate overall confidence in pattern detection"""
        confidence = 0.0
        
        # Weight different factors
        if patterns['repeated_sequences']:
            confidence += 0.3
        
        if patterns['derivation_method']:
            confidence += 0.4
            
        if patterns['structure_hints']:
            confidence += 0.3
            
        # Normalize to 0-1
        return min(confidence, 1.0)