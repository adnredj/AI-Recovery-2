import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from ..utils.logging import setup_logger
from datetime import datetime

class FeatureExtractor:
    """Extract features from Bitcoin wallet data"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.feature_dims = self._initialize_feature_dimensions()
        
    def _initialize_feature_dimensions(self) -> Dict[str, int]:
        """Initialize feature dimensions based on configuration
        
        Returns:
            Dictionary containing feature dimensions
        """
        return {
            'structure': self.config.get('structure_dim', 64),
            'version': self.config.get('version_dim', 32),
            'transaction': self.config.get('transaction_dim', 48),
            'address': self.config.get('address_dim', 32),
            'metadata': self.config.get('metadata_dim', 16)
        }
        
    def extract_features(self, wallet_data: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """Extract all features from wallet data
        
        Args:
            wallet_data: Dictionary containing wallet data
            
        Returns:
            Dictionary containing feature tensors
        """
        try:
            features = {
                'structure': self._extract_structure_features(wallet_data),
                'version': self._extract_version_features(wallet_data.get('header', {})),
                'transaction': self._extract_transaction_features(wallet_data),
                'address': self._extract_address_features(wallet_data),
                'metadata': self._extract_metadata_features(wallet_data)
            }
            
            # Combine features if specified
            if self.config.get('combine_features', False):
                features['combined'] = self._combine_features(features)
                
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return self._get_zero_features()
            
    def _get_zero_features(self) -> Dict[str, torch.Tensor]:
        """Get dictionary of zero tensors with correct dimensions"""
        return {
            name: torch.zeros(dim) 
            for name, dim in self.feature_dims.items()
        }
        
    def _combine_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine individual feature tensors
        
        Args:
            features: Dictionary of feature tensors
            
        Returns:
            Combined feature tensor
        """
        feature_list = []
        for name in ['structure', 'version', 'transaction', 'address', 'metadata']:
            if name in features:
                feature_list.append(features[name])
                
        if feature_list:
            return torch.cat(feature_list)
        return torch.zeros(sum(self.feature_dims.values()))
        
    def _extract_structure_features(self, wallet_data: Dict[str, any]) -> torch.Tensor:
        """Extract structural features from wallet data
        
        Args:
            wallet_data: Dictionary containing wallet data
            
        Returns:
            Tensor containing structural features
        """
        try:
            # Initialize feature vector
            features = []
            
            # Extract basic structure features
            basic_features = self._extract_basic_structure(wallet_data)
            features.extend(basic_features)
            
            # Extract key hierarchy features
            hierarchy_features = self._extract_hierarchy_features(wallet_data)
            features.extend(hierarchy_features)
            
            # Extract transaction structure features
            tx_features = self._extract_transaction_structure(wallet_data)
            features.extend(tx_features)
            
            # Extract address features
            addr_features = self._extract_address_features(wallet_data)
            features.extend(addr_features)
            
            # Extract metadata features
            meta_features = self._extract_metadata_structure(wallet_data)
            features.extend(meta_features)
            
            # Convert to tensor and normalize
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            normalized_features = self._normalize_features(feature_tensor)
            
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Error extracting structure features: {str(e)}")
            # Return zero tensor with correct dimensions
            return torch.zeros(self.feature_dims['structure'])
            
    def _extract_version_features(self, header: Dict[str, any]) -> torch.Tensor:
        """Extract version-specific features from wallet header
        
        Args:
            header: Dictionary containing wallet header data
            
        Returns:
            Tensor containing version features
        """
        try:
            # Initialize feature vector
            features = []
            
            # Extract version number features
            version_num = self._parse_version_number(header.get('version', ''))
            features.extend(version_num)
            
            # Extract format features
            format_features = self._extract_format_features(header)
            features.extend(format_features)
            
            # Extract compatibility features
            compat_features = self._extract_compatibility_features(header)
            features.extend(compat_features)
            
            # Extract timestamp features
            time_features = self._extract_timestamp_features(header)
            features.extend(time_features)
            
            # Extract feature flags
            flag_features = self._extract_feature_flags(header)
            features.extend(flag_features)
            
            # Convert to tensor and normalize
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            normalized_features = self._normalize_features(feature_tensor)
            
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Error extracting version features: {str(e)}")
            # Return zero tensor with correct dimensions
            return torch.zeros(self.feature_dims['version'])
            
    def _extract_basic_structure(self, wallet_data: Dict[str, any]) -> List[float]:
        """Extract basic structural features"""
        features = []
        
        # Size-based features
        features.append(float(len(str(wallet_data))))  # Total size
        features.append(float(len(wallet_data.keys())))  # Number of top-level keys
        
        # Depth features
        max_depth = self._calculate_structure_depth(wallet_data)
        features.append(float(max_depth))
        
        # Complexity features
        complexity = self._calculate_structure_complexity(wallet_data)
        features.append(complexity)
        
        return features
        
    def _extract_hierarchy_features(self, wallet_data: Dict[str, any]) -> List[float]:
        """Extract key hierarchy features"""
        features = []
        
        # Key counts
        key_info = self._analyze_key_hierarchy(wallet_data)
        features.append(float(key_info['total_keys']))
        features.append(float(key_info['master_keys']))
        features.append(float(key_info['derived_keys']))
        features.append(float(key_info['max_derivation_depth']))
        
        # Hierarchy structure
        features.append(float(key_info['branching_factor']))
        features.append(float(key_info['leaf_ratio']))
        
        return features
        
    def _extract_transaction_structure(self, wallet_data: Dict[str, any]) -> List[float]:
        """Extract transaction structure features"""
        features = []
        
        tx_data = wallet_data.get('transactions', {})
        
        # Transaction counts
        features.append(float(len(tx_data)))
        
        if tx_data:
            # Transaction complexity
            avg_inputs = np.mean([len(tx.get('inputs', [])) for tx in tx_data.values()])
            avg_outputs = np.mean([len(tx.get('outputs', [])) for tx in tx_data.values()])
            features.extend([avg_inputs, avg_outputs])
            
            # Script types
            script_types = self._analyze_script_types(tx_data)
            features.extend(list(script_types.values()))
        else:
            # Padding for missing transaction data
            features.extend([0.0] * 5)  # Adjust padding size as needed
            
        return features
        
    def _extract_address_features(self, wallet_data: Dict[str, any]) -> List[float]:
        """Extract address-related features"""
        features = []
        
        addresses = wallet_data.get('addresses', {})
        
        # Address counts
        features.append(float(len(addresses)))
        
        if addresses:
            # Address types
            addr_types = self._analyze_address_types(addresses)
            features.extend(list(addr_types.values()))
            
            # Address usage
            usage_stats = self._analyze_address_usage(addresses)
            features.extend([
                usage_stats['active_ratio'],
                usage_stats['reuse_ratio']
            ])
        else:
            # Padding for missing address data
            features.extend([0.0] * 6)  # Adjust padding size as needed
            
        return features
        
    def _extract_metadata_structure(self, wallet_data: Dict[str, any]) -> List[float]:
        """Extract metadata structure features"""
        features = []
        
        metadata = wallet_data.get('metadata', {})
        
        # Metadata completeness
        completeness = self._calculate_metadata_completeness(metadata)
        features.append(completeness)
        
        # Custom fields
        custom_fields = float(len(metadata.get('custom', {})))
        features.append(custom_fields)
        
        # Labels and tags
        label_stats = self._analyze_labels(metadata)
        features.extend([
            label_stats['label_ratio'],
            label_stats['tag_count']
        ])
        
        return features
        
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize feature tensor"""
        # Apply feature-specific normalization
        if len(features) > 0:
            mean = features.mean()
            std = features.std()
            if std > 0:
                return (features - mean) / std
        return features

    def _calculate_structure_depth(self, data: Dict[str, any], current_depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary structure
        
        Args:
            data: Dictionary to analyze
            current_depth: Current depth in recursion
            
        Returns:
            Maximum depth of structure
        """
        if not isinstance(data, dict) or not data:
            return current_depth
            
        depths = [current_depth]  # Initialize with current depth
        
        for value in data.values():
            if isinstance(value, dict):
                depths.append(self._calculate_structure_depth(value, current_depth + 1))
                
        return max(depths)  # This will now always have at least one value

    def _parse_version_number(self, version: str) -> List[float]:
        """Parse version string into feature vector
        
        Args:
            version: Version string
            
        Returns:
            List of version features
        """
        try:
            # Handle empty version
            if not version:
                return [0.0] * 3
                
            # Split version string
            parts = str(version).split('.')
            
            # Convert to floats and normalize
            features = [float(p) / 100.0 for p in parts[:3]]
            
            # Pad if needed
            while len(features) < 3:
                features.append(0.0)
                
            return features
            
        except Exception:
            return [0.0] * 3

    def _extract_transaction_features(self, wallet_data: Dict[str, any]) -> torch.Tensor:
        """Extract transaction-related features
        
        Args:
            wallet_data: Wallet data dictionary
            
        Returns:
            Transaction feature tensor
        """
        try:
            features = []
            
            # Get transactions
            transactions = wallet_data.get('transactions', {})
            
            # Basic transaction stats
            features.append(float(len(transactions)))
            
            if transactions:
                # Transaction type distribution
                type_dist = self._get_transaction_type_distribution(transactions)
                features.extend(list(type_dist.values()))
                
                # Input/output stats
                io_stats = self._get_transaction_io_stats(transactions)
                features.extend([
                    io_stats['avg_inputs'],
                    io_stats['avg_outputs'],
                    io_stats['max_inputs'],
                    io_stats['max_outputs']
                ])
                
                # Value stats
                value_stats = self._get_transaction_value_stats(transactions)
                features.extend([
                    value_stats['avg_value'],
                    value_stats['max_value'],
                    value_stats['total_value']
                ])
            else:
                # Padding for empty transactions
                features.extend([0.0] * 10)
                
            # Convert to tensor
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Ensure correct dimension
            if len(feature_tensor) < self.feature_dims['transaction']:
                feature_tensor = torch.cat([
                    feature_tensor,
                    torch.zeros(self.feature_dims['transaction'] - len(feature_tensor))
                ])
            else:
                feature_tensor = feature_tensor[:self.feature_dims['transaction']]
                
            return self._normalize_features(feature_tensor)
            
        except Exception as e:
            self.logger.error(f"Error extracting transaction features: {str(e)}")
            return torch.zeros(self.feature_dims['transaction'])

    def _get_transaction_type_distribution(self, transactions: Dict[str, any]) -> Dict[str, float]:
        """Get distribution of transaction types
        
        Args:
            transactions: Transaction dictionary
            
        Returns:
            Dictionary of type distributions
        """
        type_counts = {
            'p2pkh': 0,
            'p2sh': 0,
            'p2wpkh': 0,
            'p2wsh': 0,
            'other': 0
        }
        
        total = len(transactions)
        if total == 0:
            return {k: 0.0 for k in type_counts}
            
        for tx in transactions.values():
            tx_type = tx.get('type', 'other')
            if tx_type in type_counts:
                type_counts[tx_type] += 1
            else:
                type_counts['other'] += 1
                
        return {k: v / total for k, v in type_counts.items()}

    def _get_transaction_io_stats(self, transactions: Dict[str, any]) -> Dict[str, float]:
        """Calculate transaction input/output statistics
        
        Args:
            transactions: Transaction dictionary
            
        Returns:
            Dictionary of I/O statistics
        """
        inputs = [len(tx.get('inputs', [])) for tx in transactions.values()]
        outputs = [len(tx.get('outputs', [])) for tx in transactions.values()]
        
        return {
            'avg_inputs': np.mean(inputs) if inputs else 0.0,
            'avg_outputs': np.mean(outputs) if outputs else 0.0,
            'max_inputs': max(inputs) if inputs else 0.0,
            'max_outputs': max(outputs) if outputs else 0.0
        }

    def _get_transaction_value_stats(self, transactions: Dict[str, any]) -> Dict[str, float]:
        """Calculate transaction value statistics
        
        Args:
            transactions: Transaction dictionary
            
        Returns:
            Dictionary of value statistics
        """
        values = []
        for tx in transactions.values():
            value = sum(out.get('value', 0.0) for out in tx.get('outputs', []))
            values.append(value)
            
        if not values:
            return {
                'avg_value': 0.0,
                'max_value': 0.0,
                'total_value': 0.0
            }
            
        return {
            'avg_value': np.mean(values),
            'max_value': max(values),
            'total_value': sum(values)
        }

    def _extract_format_features(self, header: Dict[str, any]) -> List[float]:
        """Extract format-specific features
        
        Args:
            header: Wallet header dictionary
            
        Returns:
            List of format features
        """
        features = []
        
        # Format version
        format_version = header.get('format_version', 0)
        features.append(float(format_version) / 100.0)
        
        # Format type
        format_types = {
            'raw': 0.0,
            'compressed': 0.1,
            'encrypted': 0.2,
            'hd': 0.3,
            'unknown': 0.9
        }
        format_type = header.get('format_type', 'unknown')
        features.append(format_types.get(format_type, 0.9))
        
        # Format flags
        flags = header.get('format_flags', [])
        flag_features = [
            1.0 if 'compressed' in flags else 0.0,
            1.0 if 'encrypted' in flags else 0.0,
            1.0 if 'hd' in flags else 0.0,
            1.0 if 'legacy' in flags else 0.0
        ]
        features.extend(flag_features)
        
        return features
        
    def _extract_metadata_features(self, wallet_data: Dict[str, any]) -> torch.Tensor:
        """Extract metadata features
        
        Args:
            wallet_data: Wallet data dictionary
            
        Returns:
            Metadata feature tensor
        """
        try:
            features = []
            metadata = wallet_data.get('metadata', {})
            
            # Basic metadata stats
            features.append(float(len(metadata)))
            features.append(float(len(metadata.get('labels', {}))))
            features.append(float(len(metadata.get('tags', []))))
            
            # Creation time
            creation_time = metadata.get('creation_time', 0)
            features.append(float(creation_time) / 1e9)  # Normalize timestamp
            
            # Last modified time
            modified_time = metadata.get('last_modified', 0)
            features.append(float(modified_time) / 1e9)
            
            # Feature flags
            flags = metadata.get('flags', [])
            flag_features = [
                1.0 if 'backup_enabled' in flags else 0.0,
                1.0 if 'watch_only' in flags else 0.0,
                1.0 if 'hd_enabled' in flags else 0.0
            ]
            features.extend(flag_features)
            
            # Convert to tensor
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Ensure correct dimension
            if len(feature_tensor) < self.feature_dims['metadata']:
                feature_tensor = torch.cat([
                    feature_tensor,
                    torch.zeros(self.feature_dims['metadata'] - len(feature_tensor))
                ])
            else:
                feature_tensor = feature_tensor[:self.feature_dims['metadata']]
                
            return self._normalize_features(feature_tensor)
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata features: {str(e)}")
            return torch.zeros(self.feature_dims['metadata'])

    def _calculate_structure_complexity(self, data: Dict[str, any]) -> float:
        """Calculate structural complexity score
        
        Args:
            data: Dictionary to analyze
            
        Returns:
            Complexity score between 0 and 1
        """
        try:
            # Initialize complexity metrics
            metrics = {
                'depth': 0,
                'breadth': 0,
                'leaf_count': 0,
                'total_nodes': 0,
                'type_variety': set()
            }
            
            # Calculate metrics
            self._analyze_structure(data, metrics, depth=0)
            
            # Calculate complexity score
            if metrics['total_nodes'] == 0:
                return 0.0
                
            complexity = (
                0.3 * (metrics['depth'] / 10.0) +  # Depth contribution
                0.2 * (metrics['breadth'] / 50.0) +  # Breadth contribution
                0.2 * (metrics['leaf_count'] / metrics['total_nodes']) +  # Leaf ratio
                0.3 * (len(metrics['type_variety']) / 5.0)  # Type variety
            )
            
            return min(max(complexity, 0.0), 1.0)  # Ensure range [0,1]
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {str(e)}")
            return 0.0
            
    def _analyze_structure(self, data: Any, metrics: Dict[str, any], depth: int = 0):
        """Analyze structure recursively to gather metrics
        
        Args:
            data: Data to analyze
            metrics: Metrics dictionary to update
            depth: Current depth
        """
        # Update depth
        metrics['depth'] = max(metrics['depth'], depth)
        
        # Update total nodes
        metrics['total_nodes'] += 1
        
        # Add type to variety
        metrics['type_variety'].add(type(data).__name__)
        
        if isinstance(data, dict):
            # Update breadth
            metrics['breadth'] = max(metrics['breadth'], len(data))
            
            # Recurse into dictionary
            for value in data.values():
                self._analyze_structure(value, metrics, depth + 1)
                
        elif isinstance(data, (list, tuple, set)):
            # Update breadth
            metrics['breadth'] = max(metrics['breadth'], len(data))
            
            # Recurse into sequence
            for item in data:
                self._analyze_structure(item, metrics, depth + 1)
                
        else:
            # Count leaf nodes
            metrics['leaf_count'] += 1
            
    def _extract_compatibility_features(self, header: Dict[str, any]) -> List[float]:
        """Extract compatibility-related features
        
        Args:
            header: Wallet header dictionary
            
        Returns:
            List of compatibility features
        """
        features = []
        
        # Version compatibility
        min_version = header.get('min_version', 0)
        max_version = header.get('max_version', 999999)
        features.extend([
            float(min_version) / 100.0,
            float(max_version) / 100.0
        ])
        
        # Client compatibility
        client_compat = {
            'bitcoin_core': 0.1,
            'multibit': 0.2,
            'electrum': 0.3,
            'armory': 0.4,
            'unknown': 0.9
        }
        client = header.get('client', 'unknown')
        features.append(client_compat.get(client, 0.9))
        
        # Feature compatibility
        compat_flags = header.get('compatibility_flags', [])
        feature_flags = {
            'segwit': False,
            'taproot': False,
            'script_v2': False,
            'descriptor': False,
            'bip39': False,
            'legacy': False
        }
        
        # Update flags from header
        for flag in compat_flags:
            if flag in feature_flags:
                feature_flags[flag] = True
                
        # Add flag features
        features.extend([1.0 if v else 0.0 for v in feature_flags.values()])
        
        # Add format compatibility score
        format_score = self._calculate_format_compatibility(header)
        features.append(format_score)
        
        return features
        
    def _calculate_format_compatibility(self, header: Dict[str, any]) -> float:
        """Calculate format compatibility score
        
        Args:
            header: Wallet header dictionary
            
        Returns:
            Compatibility score between 0 and 1
        """
        try:
            # Initialize score
            score = 1.0
            
            # Check format version
            format_version = header.get('format_version', 0)
            if format_version < 100:  # Very old format
                score *= 0.5
            elif format_version < 200:  # Legacy format
                score *= 0.8
                
            # Check encryption compatibility
            if header.get('encryption_type') == 'legacy':
                score *= 0.7
                
            # Check feature support
            unsupported_features = header.get('unsupported_features', [])
            if unsupported_features:
                score *= (1.0 - 0.1 * len(unsupported_features))
                
            return max(min(score, 1.0), 0.0)  # Ensure range [0,1]
            
        except Exception as e:
            self.logger.error(f"Error calculating format compatibility: {str(e)}")
            return 0.0

    def _analyze_key_hierarchy(self, wallet_data: Dict[str, any]) -> Dict[str, float]:
        """Analyze key hierarchy structure
        
        Args:
            wallet_data: Wallet data dictionary
            
        Returns:
            Dictionary containing hierarchy metrics
        """
        try:
            metrics = {
                'total_keys': 0,
                'master_keys': 0,
                'derived_keys': 0,
                'max_derivation_depth': 0,
                'branching_factor': 0,
                'leaf_ratio': 0
            }
            
            # Get key data
            keys = wallet_data.get('keys', {})
            if not keys:
                return metrics
                
            # Count key types
            metrics['total_keys'] = len(keys)
            metrics['master_keys'] = sum(1 for k in keys.values() if k.get('type') == 'master')
            metrics['derived_keys'] = sum(1 for k in keys.values() if k.get('type') == 'derived')
            
            # Analyze derivation paths
            depths = []
            children_counts = []
            leaf_count = 0
            
            for key in keys.values():
                path = key.get('derivation_path', '')
                if path:
                    depth = len(path.split('/')) - 1
                    depths.append(depth)
                    
                children = key.get('derived_keys', [])
                children_counts.append(len(children))
                
                if not children:  # Leaf node
                    leaf_count += 1
                    
            # Calculate metrics
            if depths:
                metrics['max_derivation_depth'] = max(depths)
            if children_counts:
                metrics['branching_factor'] = sum(children_counts) / len(children_counts)
            if metrics['total_keys'] > 0:
                metrics['leaf_ratio'] = leaf_count / metrics['total_keys']
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing key hierarchy: {str(e)}")
            return {k: 0.0 for k in ['total_keys', 'master_keys', 'derived_keys', 
                                   'max_derivation_depth', 'branching_factor', 'leaf_ratio']}
                                   
    def _extract_timestamp_features(self, header: Dict[str, any]) -> List[float]:
        """Extract timestamp-related features
        
        Args:
            header: Wallet header dictionary
            
        Returns:
            List of timestamp features
        """
        try:
            features = []
            current_time = datetime.now().timestamp()
            
            # Creation time
            creation_time = header.get('creation_time', 0)
            if creation_time > 0:
                features.append(creation_time / current_time)  # Normalize
                features.append(1.0)  # Has creation time
            else:
                features.extend([0.0, 0.0])
                
            # Last modified time
            modified_time = header.get('last_modified', 0)
            if modified_time > 0:
                features.append(modified_time / current_time)  # Normalize
                features.append(1.0)  # Has modified time
            else:
                features.extend([0.0, 0.0])
                
            # Time-based flags
            flags = []
            
            # Check if wallet is old
            if creation_time > 0 and (current_time - creation_time) > (5 * 365 * 24 * 3600):  # 5 years
                flags.append(1.0)  # Old wallet
            else:
                flags.append(0.0)
                
            # Check if recently modified
            if modified_time > 0 and (current_time - modified_time) < (30 * 24 * 3600):  # 30 days
                flags.append(1.0)  # Recently modified
            else:
                flags.append(0.0)
                
            # Check if potentially corrupted timestamp
            if creation_time > current_time or modified_time > current_time:
                flags.append(1.0)  # Invalid timestamp
            else:
                flags.append(0.0)
                
            features.extend(flags)
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting timestamp features: {str(e)}")
            return [0.0] * 7  # Return zeros for all features

    def _calculate_metadata_completeness(self, metadata: Dict[str, any]) -> float:
        """Calculate metadata completeness score
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Completeness score between 0 and 1
        """
        try:
            # Define expected metadata fields
            expected_fields = {
                'version': 1.0,
                'name': 0.8,
                'description': 0.6,
                'creation_time': 1.0,
                'last_modified': 0.8,
                'label_data': 0.7,
                'key_pool': 0.9,
                'hd_seed': 1.0,
                'encryption_type': 0.9,
                'backup_time': 0.6
            }
            
            if not metadata:
                return 0.0
                
            # Calculate weighted completeness
            total_weight = sum(expected_fields.values())
            current_weight = sum(
                weight for field, weight in expected_fields.items()
                if field in metadata and metadata[field] is not None
            )
            
            return current_weight / total_weight
            
        except Exception as e:
            self.logger.error(f"Error calculating metadata completeness: {str(e)}")
            return 0.0
            
    def _extract_feature_flags(self, header: Dict[str, any]) -> List[float]:
        """Extract feature flag information
        
        Args:
            header: Wallet header dictionary
            
        Returns:
            List of feature flag values
        """
        try:
            # Define feature flags to check
            feature_flags = {
                'hd_enabled': False,
                'encryption_enabled': False,
                'compression_enabled': False,
                'script_verification': False,
                'key_pool_enabled': False,
                'watch_only_enabled': False,
                'multi_sig_enabled': False,
                'descriptor_enabled': False
            }
            
            # Check flags in header
            flags = header.get('feature_flags', [])
            for flag in flags:
                if flag in feature_flags:
                    feature_flags[flag] = True
                    
            # Additional feature detection
            if header.get('hd_seed'):
                feature_flags['hd_enabled'] = True
            if header.get('encryption_type'):
                feature_flags['encryption_enabled'] = True
            if header.get('key_pool'):
                feature_flags['key_pool_enabled'] = True
                
            # Convert to float values
            return [1.0 if v else 0.0 for v in feature_flags.values()]
            
        except Exception as e:
            self.logger.error(f"Error extracting feature flags: {str(e)}")
            return [0.0] * len(feature_flags)

    def _analyze_labels(self, metadata: Dict[str, any]) -> Dict[str, float]:
        """Analyze label and tag information from metadata
        
        Args:
            metadata: Metadata dictionary containing labels and tags
            
        Returns:
            Dictionary containing label analysis metrics
        """
        try:
            # Initialize metrics
            metrics = {
                'label_ratio': 0.0,
                'tag_count': 0.0,
                'label_complexity': 0.0,
                'unique_tags': 0.0,
                'hierarchical_depth': 0.0
            }
            
            # Get labels and tags
            labels = metadata.get('labels', {})
            tags = metadata.get('tags', [])
            
            if not labels and not tags:
                return metrics
                
            # Analyze labels
            if labels:
                # Count labeled items
                total_items = sum(1 for v in metadata.values() if isinstance(v, dict))
                labeled_items = len(labels)
                metrics['label_ratio'] = labeled_items / max(total_items, 1)
                
                # Analyze label complexity
                label_lengths = [len(str(label)) for label in labels.values()]
                if label_lengths:
                    metrics['label_complexity'] = sum(label_lengths) / len(label_lengths) / 100.0
                    
                # Check for hierarchical labels
                hierarchical = [label for label in labels.values() 
                              if isinstance(label, str) and '/' in label]
                if hierarchical:
                    depths = [len(label.split('/')) for label in hierarchical]
                    metrics['hierarchical_depth'] = max(depths)
                    
            # Analyze tags
            if tags:
                metrics['tag_count'] = len(tags)
                metrics['unique_tags'] = len(set(tags))
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing labels: {str(e)}")
            return {
                'label_ratio': 0.0,
                'tag_count': 0.0,
                'label_complexity': 0.0,
                'unique_tags': 0.0,
                'hierarchical_depth': 0.0
            }