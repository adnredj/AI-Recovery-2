from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
from pathlib import Path
from ..utils.logging import setup_logger

class DataCleaner:
    """Data cleaning and preprocessing for bitcoin wallet recovery"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.cleaning_stats = {
            'total_processed': 0,
            'invalid_removed': 0,
            'duplicates_removed': 0,
            'anomalies_detected': 0,
            'data_fixed': 0
        }
        self.known_patterns = self._load_known_patterns()
        
    def clean_wallet_data(self,
                         wallet_data: Dict[str, any],
                         validate: bool = True) -> Dict[str, any]:
        """Clean and validate wallet data"""
        self.logger.info("Starting wallet data cleaning")
        
        try:
            # Validate basic structure
            if validate:
                self._validate_wallet_structure(wallet_data)
            
            # Clean individual components
            cleaned_data = {
                'metadata': self._clean_metadata(wallet_data.get('metadata', {})),
                'keys': self._clean_keys(wallet_data.get('keys', [])),
                'addresses': self._clean_addresses(wallet_data.get('addresses', [])),
                'transactions': self._clean_transactions(wallet_data.get('transactions', [])),
                'scripts': self._clean_scripts(wallet_data.get('scripts', []))
            }
            
            # Update cleaning stats
            self.cleaning_stats['total_processed'] += 1
            
            # Validate cleaned data
            if validate:
                self._validate_cleaned_data(cleaned_data)
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error cleaning wallet data: {str(e)}")
            raise
    
    def clean_transaction_batch(self,
                              transactions: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Clean a batch of transactions"""
        cleaned_transactions = []
        
        for tx in transactions:
            try:
                cleaned_tx = self._clean_single_transaction(tx)
                if cleaned_tx:
                    cleaned_transactions.append(cleaned_tx)
            except Exception as e:
                self.logger.warning(f"Error cleaning transaction: {str(e)}")
                self.cleaning_stats['invalid_removed'] += 1
                
        return cleaned_transactions
    
    def clean_address_batch(self,
                          addresses: List[str]) -> List[str]:
        """Clean a batch of bitcoin addresses"""
        cleaned_addresses = []
        seen_addresses = set()
        
        for addr in addresses:
            try:
                cleaned_addr = self._clean_single_address(addr)
                if cleaned_addr and cleaned_addr not in seen_addresses:
                    cleaned_addresses.append(cleaned_addr)
                    seen_addresses.add(cleaned_addr)
                else:
                    self.cleaning_stats['duplicates_removed'] += 1
            except Exception as e:
                self.logger.warning(f"Error cleaning address: {str(e)}")
                self.cleaning_stats['invalid_removed'] += 1
                
        return cleaned_addresses
    
    def _clean_metadata(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """Clean wallet metadata"""
        cleaned_metadata = {}
        
        # Clean timestamp
        if 'creation_time' in metadata:
            cleaned_metadata['creation_time'] = self._clean_timestamp(
                metadata['creation_time']
            )
            
        # Clean version info
        if 'version' in metadata:
            cleaned_metadata['version'] = self._clean_version(
                metadata['version']
            )
            
        # Clean labels and descriptions
        if 'labels' in metadata:
            cleaned_metadata['labels'] = self._clean_text_fields(
                metadata['labels']
            )
            
        return cleaned_metadata
    
    def _clean_keys(self, keys: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Clean wallet keys"""
        cleaned_keys = []
        seen_keys = set()
        
        for key in keys:
            try:
                # Clean key data
                cleaned_key = self._clean_single_key(key)
                
                # Check for duplicates
                key_hash = self._hash_key(cleaned_key)
                if key_hash not in seen_keys:
                    cleaned_keys.append(cleaned_key)
                    seen_keys.add(key_hash)
                else:
                    self.cleaning_stats['duplicates_removed'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Error cleaning key: {str(e)}")
                self.cleaning_stats['invalid_removed'] += 1
                
        return cleaned_keys
    
    def _clean_single_transaction(self,
                                transaction: Dict[str, any]) -> Optional[Dict[str, any]]:
        """Clean single transaction data"""
        try:
            cleaned_tx = {}
            
            # Clean basic fields
            cleaned_tx['txid'] = self._clean_txid(transaction.get('txid', ''))
            cleaned_tx['timestamp'] = self._clean_timestamp(transaction.get('timestamp'))
            
            # Clean amounts
            cleaned_tx['amount'] = self._clean_amount(transaction.get('amount', 0))
            cleaned_tx['fee'] = self._clean_amount(transaction.get('fee', 0))
            
            # Clean inputs and outputs
            cleaned_tx['inputs'] = self._clean_tx_components(transaction.get('inputs', []))
            cleaned_tx['outputs'] = self._clean_tx_components(transaction.get('outputs', []))
            
            # Validate cleaned transaction
            if self._is_valid_transaction(cleaned_tx):
                return cleaned_tx
            else:
                self.cleaning_stats['invalid_removed'] += 1
                return None
                
        except Exception as e:
            self.logger.warning(f"Error in transaction cleaning: {str(e)}")
            return None
    
    def _clean_single_address(self, address: str) -> Optional[str]:
        """Clean single bitcoin address"""
        if not address:
            return None
            
        # Remove whitespace
        cleaned = address.strip()
        
        # Validate basic format
        if not self._is_valid_address_format(cleaned):
            return None
            
        # Fix common errors
        cleaned = self._fix_common_address_errors(cleaned)
        
        # Validate final address
        if self._is_valid_address(cleaned):
            return cleaned
        else:
            return None
    
    def _clean_scripts(self, scripts: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Clean script data"""
        cleaned_scripts = []
        
        for script in scripts:
            try:
                cleaned_script = self._clean_single_script(script)
                if cleaned_script:
                    cleaned_scripts.append(cleaned_script)
            except Exception as e:
                self.logger.warning(f"Error cleaning script: {str(e)}")
                self.cleaning_stats['invalid_removed'] += 1
                
        return cleaned_scripts
    
    def _clean_timestamp(self, timestamp: Union[int, str, datetime]) -> Optional[int]:
        """Clean and validate timestamp"""
        try:
            if isinstance(timestamp, datetime):
                return int(timestamp.timestamp())
            elif isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
                return int(dt.timestamp())
            elif isinstance(timestamp, (int, float)):
                return int(timestamp)
            else:
                return None
        except:
            return None
    
    def _clean_amount(self, amount: Union[int, float, str]) -> float:
        """Clean and validate bitcoin amount"""
        try:
            if isinstance(amount, str):
                amount = float(amount.replace(',', ''))
            return max(0.0, float(amount))
        except:
            return 0.0
    
    def _clean_text_fields(self, text_data: Union[str, Dict, List]) -> Union[str, Dict, List]:
        """Clean text fields"""
        if isinstance(text_data, str):
            return text_data.strip()
        elif isinstance(text_data, dict):
            return {k: self._clean_text_fields(v) for k, v in text_data.items()}
        elif isinstance(text_data, list):
            return [self._clean_text_fields(item) for item in text_data]
        return text_data
    
    def _validate_wallet_structure(self, wallet_data: Dict[str, any]):
        """Validate wallet data structure"""
        required_fields = ['version', 'type', 'created_at']
        for field in required_fields:
            if field not in wallet_data:
                raise ValueError(f"Missing required field: {field}")
    
    def _validate_cleaned_data(self, cleaned_data: Dict[str, any]):
        """Validate cleaned data"""
        # Validate relationships between components
        self._validate_key_address_relationships(
            cleaned_data['keys'],
            cleaned_data['addresses']
        )
        
        # Validate transaction consistency
        self._validate_transaction_consistency(
            cleaned_data['transactions']
        )
    
    def _load_known_patterns(self) -> Dict[str, any]:
        """Load known patterns for data cleaning"""
        pattern_file = Path(self.config.pattern_file)
        if pattern_file.exists():
            # Load patterns from file
            pass
        return {}
    
    def get_cleaning_stats(self) -> Dict[str, int]:
        """Get data cleaning statistics"""
        return self.cleaning_stats.copy()