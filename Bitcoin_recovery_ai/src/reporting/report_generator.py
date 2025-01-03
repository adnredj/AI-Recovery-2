import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
from collections import OrderedDict
from ..utils.logging import setup_logger

class RecoveryAnalysisReport:
    """Generate detailed analysis reports for wallet recovery attempts"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.report_dir = Path('reports')
        self.report_dir.mkdir(exist_ok=True)
        self.version_patterns = self._load_version_patterns()
        
    def generate_report(self,
                       wallet_data: Dict[str, any],
                       predictions: Dict[str, any],
                       recovery_results: List[Dict[str, any]]) -> Dict[str, any]:
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report = {
            'timestamp': timestamp,
            'wallet_analysis': self._analyze_wallet_structure(wallet_data),
            'encryption_analysis': self._analyze_encryption(wallet_data, predictions),
            'recovery_analysis': self._analyze_recovery_results(recovery_results),
            'recommendations': self._generate_recommendations(wallet_data, predictions),
            'confidence_metrics': self._calculate_confidence_metrics(predictions)
        }
        
        # Save report
        report_path = self.report_dir / f'recovery_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info(f"Generated analysis report: {report_path}")
        return report
    
    def _analyze_wallet_structure(self, wallet_data: Dict[str, any]) -> Dict[str, any]:
        """Analyze wallet file structure"""
        return {
            'version': self._detect_wallet_version(wallet_data),
            'structure_type': self._analyze_structure_type(wallet_data),
            'key_storage': {
                'type': self._identify_key_storage_type(wallet_data),
                'encryption_method': self._identify_encryption_method(wallet_data),
                'key_derivation': self._analyze_key_derivation(wallet_data)
            },
            'metadata': self._extract_metadata(wallet_data)
        }
    
    def _analyze_encryption(self,
                          wallet_data: Dict[str, any],
                          predictions: Dict[str, any]) -> Dict[str, any]:
        """Analyze encryption characteristics"""
        return {
            'encryption_type': {
                'detected': self._detect_encryption_type(wallet_data),
                'confidence': float(predictions['encryption_confidence'])
            },
            'encryption_strength': self._analyze_encryption_strength(wallet_data),
            'potential_vulnerabilities': self._identify_vulnerabilities(wallet_data),
            'pattern_analysis': {
                'detected_patterns': self._analyze_patterns(wallet_data),
                'pattern_confidence': float(predictions['pattern_confidence'])
            }
        }
    
    def _analyze_recovery_results(self, 
                                recovery_results: List[Dict[str, any]]) -> Dict[str, any]:
        """Analyze recovery attempt results"""
        successful_attempts = [r for r in recovery_results if r['success']]
        failed_attempts = [r for r in recovery_results if not r['success']]
        
        return {
            'summary': {
                'total_attempts': len(recovery_results),
                'successful_attempts': len(successful_attempts),
                'failed_attempts': len(failed_attempts),
                'success_rate': len(successful_attempts) / len(recovery_results) if recovery_results else 0
            },
            'successful_strategies': [
                {
                    'strategy': result['strategy'],
                    'confidence': result['confidence'],
                    'recovery_method': result.get('recovery_method', 'unknown')
                }
                for result in successful_attempts
            ],
            'failure_analysis': self._analyze_failures(failed_attempts)
        }
    
    def _generate_recommendations(self,
                                wallet_data: Dict[str, any],
                                predictions: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate recovery recommendations"""
        recommendations = []
        
        # Version-specific recommendations
        version_recs = self._get_version_recommendations(wallet_data)
        if version_recs:
            recommendations.extend(version_recs)
        
        # Encryption-specific recommendations
        encryption_recs = self._get_encryption_recommendations(
            wallet_data,
            predictions['encryption_pattern']
        )
        if encryption_recs:
            recommendations.extend(encryption_recs)
        
        # Pattern-based recommendations
        if float(predictions['pattern_confidence']) > self.config.thresholds.pattern_confidence:
            pattern_recs = self._get_pattern_recommendations(predictions['encryption_pattern'])
            if pattern_recs:
                recommendations.extend(pattern_recs)
        
        return recommendations
    
    def _calculate_confidence_metrics(self,
                                   predictions: Dict[str, any]) -> Dict[str, float]:
        """Calculate confidence metrics for predictions"""
        return {
            'overall_confidence': float(predictions['recovery_probability']),
            'pattern_confidence': float(predictions['pattern_confidence']),
            'version_confidence': float(predictions['version_confidence']),
            'strategy_confidence': {
                strategy: float(score)
                for strategy, score in predictions['strategy_scores'].items()
            }
        }
    
    def _detect_wallet_version(self, wallet_data: Dict[str, any]) -> str:
        """Detect wallet version from data structure and patterns"""
        try:
            # Check explicit version field
            if 'version' in wallet_data:
                return str(wallet_data['version'])
                
            # Check header magic bytes
            if 'header' in wallet_data:
                header = wallet_data['header']
                for version, pattern in self.version_patterns.items():
                    if header.startswith(pattern):
                        return version
                        
            # Check structure patterns
            if 'berkeley_db' in str(wallet_data).lower():
                return "pre-0.8.0"
            elif 'sqlite' in str(wallet_data).lower():
                return "post-0.8.0"
                
            # Check encryption method
            if 'encrypted_key' in wallet_data:
                if 'method' in wallet_data['encrypted_key']:
                    method = wallet_data['encrypted_key']['method']
                    if 'aes' in method.lower():
                        return "post-0.4.0"
                    else:
                        return "pre-0.4.0"
                        
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error detecting wallet version: {str(e)}")
            return "error"
    
    def _analyze_structure_type(self, wallet_data: Dict[str, any]) -> str:
        """Analyze wallet structure type"""
        try:
            # Check file structure indicators
            if 'berkeley_db' in str(wallet_data).lower():
                return "berkeley_db"
            elif 'sqlite' in str(wallet_data).lower():
                return "sqlite"
            elif isinstance(wallet_data, dict) and 'keys' in wallet_data:
                return "json"
            elif isinstance(wallet_data, bytes):
                return "binary"
                
            # Check specific structure patterns
            if self._has_key_hierarchy(wallet_data):
                return "hierarchical"
            elif self._has_flat_structure(wallet_data):
                return "flat"
                
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error analyzing structure: {str(e)}")
            return "error"
    
    def _identify_key_storage_type(self, wallet_data: Dict[str, any]) -> str:
        """Identify key storage method"""
        try:
            if 'master_key' in wallet_data:
                return "hierarchical_deterministic"
            elif 'key_pool' in wallet_data:
                return "key_pool"
            elif 'encrypted_key' in wallet_data:
                return "encrypted"
            elif 'keys' in wallet_data and isinstance(wallet_data['keys'], list):
                return "individual"
            elif 'keys' in wallet_data and isinstance(wallet_data['keys'], dict):
                return "mapped"
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error identifying key storage: {str(e)}")
            return "error"
    
    def _identify_encryption_method(self, wallet_data: Dict[str, any]) -> str:
        """Identify encryption method used"""
        try:
            if 'encrypted_key' not in wallet_data:
                return "none"
                
            enc_data = wallet_data['encrypted_key']
            
            # Check explicit method
            if 'method' in enc_data:
                return enc_data['method']
                
            # Check for AES indicators
            if 'iv' in enc_data and len(enc_data.get('key', '')) in [16, 24, 32]:
                return "aes-256-cbc"
                
            # Check for early encryption
            if 'salt' in enc_data and not 'iv' in enc_data:
                return "early_bitcoin"
                
            # Check for custom encryption
            if 'custom_method' in enc_data:
                return enc_data['custom_method']
                
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error identifying encryption: {str(e)}")
            return "error"
    
    def _analyze_key_derivation(self, wallet_data: Dict[str, any]) -> Dict[str, any]:
        """Analyze key derivation method"""
        try:
            result = {
                'method': 'unknown',
                'parameters': {},
                'confidence': 0.0
            }
            
            if 'key_derivation' not in wallet_data:
                return result
                
            kd_data = wallet_data['key_derivation']
            
            # Check for known methods
            if 'method' in kd_data:
                result['method'] = kd_data['method']
                result['parameters'] = kd_data.get('parameters', {})
                result['confidence'] = 1.0
                
            # Analyze parameters
            elif 'salt' in kd_data and 'iterations' in kd_data:
                result['method'] = 'pbkdf2'
                result['parameters'] = {
                    'iterations': kd_data['iterations'],
                    'salt': kd_data['salt']
                }
                result['confidence'] = 0.9
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing key derivation: {str(e)}")
            return {'method': 'error', 'parameters': {}, 'confidence': 0.0}
    
    def _extract_metadata(self, wallet_data: Dict[str, any]) -> Dict[str, any]:
        """Extract wallet metadata"""
        try:
            metadata = {
                'creation_time': None,
                'last_updated': None,
                'key_count': 0,
                'transaction_count': 0,
                'address_count': 0,
                'labels': {},
                'custom_fields': {}
            }
            
            # Extract timestamps
            if 'creation_time' in wallet_data:
                metadata['creation_time'] = datetime.fromtimestamp(
                    wallet_data['creation_time']
                )
                
            if 'last_updated' in wallet_data:
                metadata['last_updated'] = datetime.fromtimestamp(
                    wallet_data['last_updated']
                )
                
            # Count elements
            if 'keys' in wallet_data:
                metadata['key_count'] = len(wallet_data['keys'])
                
            if 'transactions' in wallet_data:
                metadata['transaction_count'] = len(wallet_data['transactions'])
                
            if 'addresses' in wallet_data:
                metadata['address_count'] = len(wallet_data['addresses'])
                
            # Extract labels and custom fields
            metadata['labels'] = wallet_data.get('labels', {})
            metadata['custom_fields'] = wallet_data.get('custom_fields', {})
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {}

    def _has_key_hierarchy(self, wallet_data: Dict[str, any]) -> bool:
        """Check if wallet has hierarchical key structure"""
        return ('master_key' in wallet_data or 
                'key_hierarchy' in wallet_data or 
                'hd_keychain' in str(wallet_data).lower())

    def _has_flat_structure(self, wallet_data: Dict[str, any]) -> bool:
        """Check if wallet has flat key structure"""
        return ('keys' in wallet_data and 
                isinstance(wallet_data['keys'], (list, dict)) and 
                not self._has_key_hierarchy(wallet_data))