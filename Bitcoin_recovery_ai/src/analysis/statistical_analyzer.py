import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from ..utils.crypto_utils import CryptoUtils
from ..utils.logging import setup_logger

class StatisticalAnalyzer:
    """Advanced statistical analysis for encryption detection"""
    
    def __init__(self, config):
        self.config = config
        self.crypto_utils = CryptoUtils()
        self.logger = setup_logger(__name__)
        
    def analyze_data(self, data: bytes) -> Dict[str, any]:
        """Perform comprehensive statistical analysis"""
        return {
            'basic_stats': self._calculate_basic_stats(data),
            'randomness_tests': self._perform_randomness_tests(data),
            'distribution_analysis': self._analyze_distribution(data),
            'correlation_analysis': self._analyze_correlations(data),
            'pattern_metrics': self._calculate_pattern_metrics(data)
        }
    
    def _calculate_basic_stats(self, data: bytes) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        return {
            'mean': float(np.mean(byte_array)),
            'median': float(np.median(byte_array)),
            'std_dev': float(np.std(byte_array)),
            'variance': float(np.var(byte_array)),
            'skewness': float(stats.skew(byte_array)),
            'kurtosis': float(stats.kurtosis(byte_array)),
            'unique_bytes': len(np.unique(byte_array)),
            'entropy': float(stats.entropy(np.bincount(byte_array)))
        }
    
    def _perform_randomness_tests(self, data: bytes) -> Dict[str, any]:
        """Perform statistical randomness tests"""
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        tests = {
            'runs_test': self._runs_test(byte_array),
            'monobit_test': self._monobit_test(byte_array),
            'chi_square_test': self._chi_square_test(byte_array),
            'serial_test': self._serial_test(byte_array),
            'longest_run_test': self._longest_run_test(byte_array)
        }
        
        # Aggregate test results
        return {
            'test_results': tests,
            'overall_randomness': self._calculate_overall_randomness(tests),
            'recommendations': self._generate_randomness_recommendations(tests)
        }
    
    def _analyze_distribution(self, data: bytes) -> Dict[str, any]:
        """Analyze byte value distribution"""
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        # Calculate histogram
        hist, bins = np.histogram(byte_array, bins=256, range=(0, 256))
        
        # Perform distribution tests
        normal_test = stats.normaltest(byte_array)
        ks_test = stats.kstest(byte_array, 'uniform', args=(0, 256))
        
        return {
            'histogram': {
                'counts': hist.tolist(),
                'bins': bins.tolist()
            },
            'distribution_tests': {
                'normal_test': {
                    'statistic': float(normal_test.statistic),
                    'p_value': float(normal_test.pvalue)
                },
                'ks_test': {
                    'statistic': float(ks_test.statistic),
                    'p_value': float(ks_test.pvalue)
                }
            },
            'uniformity_score': self._calculate_uniformity_score(hist)
        }
    
    def _analyze_correlations(self, data: bytes) -> Dict[str, any]:
        """Analyze byte-level correlations"""
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        # Calculate autocorrelation
        autocorr = self._calculate_autocorrelation(byte_array)
        
        # Calculate cross-correlation between different parts
        cross_corr = self._calculate_cross_correlation(byte_array)
        
        return {
            'autocorrelation': {
                'values': autocorr.tolist(),
                'significant_lags': self._find_significant_lags(autocorr)
            },
            'cross_correlation': {
                'matrix': cross_corr.tolist(),
                'significant_pairs': self._find_significant_correlations(cross_corr)
            },
            'correlation_summary': self._summarize_correlations(autocorr, cross_corr)
        }
    
    def _calculate_pattern_metrics(self, data: bytes) -> Dict[str, any]:
        """Calculate metrics for pattern detection"""
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        return {
            'repetition_score': self._calculate_repetition_score(byte_array),
            'sequence_metrics': self._analyze_sequences(byte_array),
            'block_similarity': self._analyze_block_similarity(byte_array),
            'pattern_summary': self._summarize_patterns(byte_array)
        }
    
    def _runs_test(self, data: np.ndarray) -> Dict[str, any]:
        """Perform runs test for randomness"""
        median = np.median(data)
        runs = np.diff(data > median)
        num_runs = np.sum(np.abs(runs)) + 1
        
        # Calculate expected number of runs
        n1 = np.sum(data > median)
        n2 = len(data) - n1
        exp_runs = ((2 * n1 * n2) / len(data)) + 1
        var_runs = ((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
                   (len(data)**2 * (len(data) - 1)))
        
        z_stat = (num_runs - exp_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'statistic': float(z_stat),
            'p_value': float(p_value),
            'num_runs': int(num_runs),
            'expected_runs': float(exp_runs)
        }
    
    def _monobit_test(self, data: np.ndarray) -> Dict[str, any]:
        """Perform monobit test for randomness"""
        # Convert to bits and count ones
        bits = np.unpackbits(data.astype(np.uint8))
        count_ones = np.sum(bits)
        
        # Calculate test statistic
        s_obs = abs(count_ones - len(bits)/2) / np.sqrt(len(bits)/4)
        p_value = 2 * (1 - stats.norm.cdf(s_obs))
        
        return {
            'statistic': float(s_obs),
            'p_value': float(p_value),
            'proportion_ones': float(count_ones / len(bits))
        }
    
    def _chi_square_test(self, data: np.ndarray) -> Dict[str, any]:
        """Perform chi-square test for uniformity"""
        observed = np.bincount(data, minlength=256)
        expected = len(data) / 256 * np.ones(256)
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        return {
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': 255
        }
    
    def _calculate_uniformity_score(self, histogram: np.ndarray) -> float:
        """Calculate uniformity score from histogram"""
        expected = np.mean(histogram)
        deviations = np.abs(histogram - expected)
        max_deviation = np.max(deviations)
        
        return 1.0 - (max_deviation / expected)
    
    def _calculate_autocorrelation(self, data: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """Calculate autocorrelation up to max_lag"""
        return np.array([np.corrcoef(data[:-i], data[i:])[0,1] 
                        for i in range(1, max_lag + 1)])
    
    def _calculate_cross_correlation(self, data: np.ndarray) -> np.ndarray:
        """Calculate cross-correlation between different parts"""
        # Split data into blocks
        block_size = len(data) // 4
        blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
        
        # Calculate correlation matrix
        n_blocks = len(blocks)
        corr_matrix = np.zeros((n_blocks, n_blocks))
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                if len(blocks[i]) == len(blocks[j]):
                    corr_matrix[i,j] = np.corrcoef(blocks[i], blocks[j])[0,1]
                    
        return corr_matrix