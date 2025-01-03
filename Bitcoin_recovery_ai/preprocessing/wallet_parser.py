import struct
from typing import Dict, Optional, Tuple, List
from datetime import datetime, MINYEAR, MAXYEAR
import logging
from pathlib import Path
import math

# Make bsddb3 optional
try:
    import bsddb3
    BSDDB_AVAILABLE = True
except ImportError:
    BSDDB_AVAILABLE = False

from cryptography.hazmat.primitives import hashes
from ..utils.logging import setup_logger

class WalletParserError(Exception):
    """Custom exception for wallet parsing errors"""
    pass

class WalletParser:
    """Parser for Bitcoin wallet.dat files"""
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not BSDDB_AVAILABLE:
            self.logger.warning(
                "bsddb3 not available. Limited wallet parsing functionality."
            )
        
    def parse_wallet(self, wallet_path: str) -> Dict[str, any]:
        """Parse wallet.dat file and extract relevant information"""
        try:
            wallet_path = Path(wallet_path)
            if not wallet_path.exists():
                raise FileNotFoundError(f"Wallet file not found: {wallet_path}")

            with open(wallet_path, 'rb') as wallet_file:
                wallet_data = wallet_file.read()
                
            # Extract wallet components
            header = self._parse_header(wallet_data[:128])
            master_key = self._extract_master_key(wallet_data)
            key_pool = self._extract_key_pool(wallet_data)
            
            # Parse Berkeley DB if available
            db_info = {}
            if BSDDB_AVAILABLE:
                try:
                    db_info = self._parse_berkeley_db(str(wallet_path))
                except Exception as e:
                    self.logger.warning(f"Berkeley DB parsing failed: {str(e)}")
                    db_info = self._parse_fallback(wallet_data)
            else:
                db_info = self._parse_fallback(wallet_data)
            
            return {
                'header': header,
                'master_key': master_key,
                'key_pool': key_pool,
                'db_info': db_info,
                'raw_data': wallet_data
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing wallet file: {str(e)}")
            raise WalletParserError(f"Wallet parsing failed: {str(e)}")

    def _parse_fallback(self, wallet_data: bytes) -> Dict[str, any]:
        """Fallback parsing method when Berkeley DB is not available"""
        try:
            return {
                'version': self._extract_version(wallet_data),
                'entries': self._extract_entries(wallet_data),
                'metadata': self._extract_metadata(wallet_data)
            }
        except Exception as e:
            self.logger.warning(f"Fallback parsing warning: {str(e)}")
            return {}

    def _extract_version(self, data: bytes) -> Optional[int]:
        """Extract wallet version from raw data"""
        try:
            version_pos = data.find(b'version')
            if version_pos != -1:
                return int.from_bytes(data[version_pos+7:version_pos+11], 'little')
        except Exception:
            pass
        return None

    def _extract_entries(self, data: bytes) -> List[Dict[str, any]]:
        """Extract wallet entries using pattern matching"""
        entries = []
        try:
            # Search for key patterns
            pos = 0
            while True:
                pos = data.find(b'key', pos + 1)
                if pos == -1:
                    break
                
                entry = self._parse_key_entry(data[pos:pos+80])
                if entry:
                    entries.append(entry)
        except Exception as e:
            self.logger.warning(f"Entry extraction warning: {str(e)}")
        return entries

    def _extract_metadata(self, data: bytes) -> Dict[str, any]:
        """Extract wallet metadata from raw data"""
        metadata = {}
        try:
            # Extract creation time
            time_pos = data.find(b'createtime')
            if time_pos != -1:
                timestamp = int.from_bytes(data[time_pos+10:time_pos+18], 'little')
                metadata['created_at'] = datetime.fromtimestamp(timestamp)
                
            # Extract label information
            label_pos = data.find(b'label')
            if label_pos != -1:
                label_len = int.from_bytes(data[label_pos+5:label_pos+6], 'little')
                label = data[label_pos+6:label_pos+6+label_len].decode('utf-8', errors='ignore')
                metadata['label'] = label
        except Exception as e:
            self.logger.warning(f"Metadata extraction warning: {str(e)}")
        return metadata

    def _parse_header(self, header_data: bytes) -> Dict[str, any]:
        """Parse wallet header information"""
        header_info = {
            'version': None,
            'encryption_type': None,
            'checksum': None,
            'features': []
        }
        
        try:
            # Check for known magic bytes
            if header_data[:4] == b'\xf9\xbe\xb4\xd9':
                header_info['version'] = struct.unpack('<I', header_data[4:8])[0]
                header_info['encryption_type'] = header_data[8]
                header_info['checksum'] = header_data[9:13]
                
                # Parse feature flags
                feature_bytes = header_data[13:17]
                header_info['features'] = self._parse_features(feature_bytes)
                
        except Exception as e:
            self.logger.warning(f"Header parsing warning: {str(e)}")
            
        return header_info
    
    def _extract_master_key(self, wallet_data: bytes) -> Optional[Dict[str, any]]:
        """Extract master key information"""
        try:
            # Search for master key markers
            master_key_pos = wallet_data.find(b'mkey')
            if master_key_pos != -1:
                # Parse master key structure
                key_data = wallet_data[master_key_pos:master_key_pos+48]
                return {
                    'encrypted_key': key_data[4:36],
                    'salt': key_data[36:44],
                    'iv': key_data[44:56] if len(key_data) >= 56 else None,
                    'method': self._identify_encryption_method(key_data)
                }
        except Exception as e:
            self.logger.warning(f"Master key extraction warning: {str(e)}")
            
        return None
    
    def _extract_key_pool(self, wallet_data: bytes) -> List[Dict[str, any]]:
        """Extract key pool information"""
        key_pool = []
        try:
            # Search for key pool markers
            pool_pos = 0
            while True:
                pool_pos = wallet_data.find(b'key', pool_pos + 1)
                if pool_pos == -1:
                    break
                    
                # Parse key entry
                key_entry = self._parse_key_entry(wallet_data[pool_pos:pool_pos+80])
                if key_entry:
                    key_pool.append(key_entry)
                    
        except Exception as e:
            self.logger.warning(f"Key pool extraction warning: {str(e)}")
            
        return key_pool
    
    def _parse_berkeley_db(self, wallet_path: str) -> Dict[str, any]:
        """Parse Berkeley DB structure"""
        db_info = {
            'version': None,
            'entries': [],
            'metadata': {}
        }
        
        try:
            # Open wallet as Berkeley DB
            db = bsddb3.btopen(wallet_path, 'r')
            
            # Extract DB version
            if b'version' in db:
                db_info['version'] = struct.unpack('<I', db[b'version'])[0]
            
            # Extract entries
            for key, value in db.items():
                entry = self._parse_db_entry(key, value)
                if entry:
                    db_info['entries'].append(entry)
                    
            db.close()
            
        except Exception as e:
            self.logger.warning(f"Berkeley DB parsing warning: {str(e)}")
            
        return db_info
    
    def _parse_features(self, feature_bytes: bytes) -> List[str]:
        """Parse wallet feature flags"""
        features = []
        if feature_bytes[0] & 0x01:
            features.append('encrypted')
        if feature_bytes[0] & 0x02:
            features.append('hd_enabled')
        if feature_bytes[0] & 0x04:
            features.append('key_pool')
        return features
    
    def _identify_encryption_method(self, key_data: bytes) -> str:
        """Identify encryption method used in wallet data
        
        Args:
            key_data: Raw key data bytes
            
        Returns:
            String identifying the encryption method
        """
        try:
            # Check minimum data length
            if len(key_data) < 16:
                return "unknown"
                
            # Extract potential header/magic bytes
            header = key_data[:16]
            
            # Check for AES-256-CBC (most common in Bitcoin Core)
            if self._check_aes_pattern(header):
                return "aes-256-cbc"
                
            # Check for Chacha20
            if self._check_chacha_pattern(header, key_data):
                return "chacha20"
                
            # Check for older encryption methods
            if self._check_legacy_encryption(header):
                return "legacy"
                
            # Check for custom encryption patterns
            custom_method = self._check_custom_patterns(key_data)
            if custom_method:
                return custom_method
                
            # Analyze entropy and block patterns
            entropy_analysis = self._analyze_encryption_entropy(key_data)
            if entropy_analysis['is_encrypted']:
                return entropy_analysis['likely_method']
                
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error identifying encryption method: {str(e)}")
            return "error"
            
    def _parse_key_entry(self, entry_data: bytes) -> Optional[Dict[str, any]]:
        """Parse individual key entry with improved error handling
        
        Args:
            entry_data: Raw key entry data
            
        Returns:
            Dictionary containing parsed key data or None if invalid
        """
        try:
            if len(entry_data) < 40:  # Minimum valid entry size
                return None
                
            result = {
                'type': None,
                'key': None,
                'metadata': {},
                'timestamp': None,
                'flags': []
            }
            
            # Parse header
            header = entry_data[:4]
            entry_type = struct.unpack('<I', header)[0]
            result['type'] = self._get_entry_type(entry_type)
            
            # Parse timestamp if present
            if len(entry_data) >= 12:
                timestamp = self._parse_timestamp(entry_data[4:12])
                if timestamp:
                    result['timestamp'] = timestamp
                    
            # Extract key data based on type
            if result['type'] == 'private':
                key_data = self._extract_private_key(entry_data[12:])
                if key_data:
                    result['key'] = key_data['key']
                    result['metadata'].update(key_data['metadata'])
                    
            elif result['type'] == 'public':
                key_data = self._extract_public_key(entry_data[12:])
                if key_data:
                    result['key'] = key_data['key']
                    result['metadata'].update(key_data['metadata'])
                    
            # Parse flags
            flags = self._parse_db_flags(entry_data[-4:])
            result['flags'] = flags
            
            # Validate entry
            if self._validate_key_entry(result):
                return result
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing key entry: {str(e)}")
            return None
            
    def _parse_db_entry(self, key: bytes, value: bytes) -> Optional[Dict[str, any]]:
        """Parse Berkeley DB entry from wallet
        
        Args:
            key: DB entry key
            value: DB entry value
            
        Returns:
            Dictionary containing parsed entry data or None if invalid
        """
        try:
            result = {
                'key_type': None,
                'data': None,
                'metadata': {},
                'created_at': None,
                'flags': []
            }
            
            # Parse key type
            key_type = self._identify_db_key_type(key)
            result['key_type'] = key_type
            
            if key_type == 'name':
                # Parse name record
                parsed = self._parse_name_record(value)
                if parsed:
                    result['data'] = parsed['name']
                    result['metadata'] = parsed['metadata']
                    
            elif key_type == 'key':
                # Parse key record
                parsed = self._parse_key_record(value)
                if parsed:
                    result['data'] = parsed['key']
                    result['metadata'] = parsed['metadata']
                    result['created_at'] = parsed.get('created_at')
                    
            elif key_type == 'pool':
                # Parse key pool entry
                parsed = self._parse_pool_entry(value)
                if parsed:
                    result['data'] = parsed['pool_data']
                    result['metadata'] = parsed['metadata']
                    
            elif key_type == 'version':
                # Parse version info
                result['data'] = int.from_bytes(value, byteorder='little')
                
            elif key_type == 'setting':
                # Parse wallet setting
                parsed = self._parse_setting_entry(key, value)
                if parsed:
                    result['data'] = parsed['value']
                    result['metadata'] = parsed['metadata']
                    
            # Parse common flags
            if len(value) >= 4:
                result['flags'] = self._parse_db_flags(value[-4:])
                
            # Validate entry
            if self._validate_db_entry(result):
                return result
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing DB entry: {str(e)}")
            return None
            
    def _check_aes_pattern(self, header: bytes) -> bool:
        """Check for AES encryption pattern"""
        # Check for common AES indicators
        return (
            len(header) == 16 and
            header.startswith(b'\x00\x02') and  # Common AES header
            sum(header[2:4]) > 0 and  # IV indicator
            header[-4:] != b'\x00' * 4  # Non-zero padding
        )
        
    def _check_chacha_pattern(self, header: bytes, data: bytes) -> bool:
        """Check for ChaCha20 encryption pattern"""
        return (
            len(header) == 16 and
            header.startswith(b'\x01\x02') and  # ChaCha header
            len(data) % 64 == 0  # ChaCha block size
        )
        
    def _analyze_encryption_entropy(self, data: bytes) -> Dict[str, any]:
        """Analyze entropy patterns to identify encryption"""
        result = {
            'is_encrypted': False,
            'likely_method': 'unknown',
            'confidence': 0.0
        }
        
        # Calculate entropy
        entropy = self._calculate_entropy(data)
        
        # High entropy indicates encryption
        if entropy > 7.8:  # Close to maximum entropy of 8
            result['is_encrypted'] = True
            result['confidence'] = min((entropy - 7.8) * 5, 1.0)
            
            # Analyze block patterns
            if len(data) % 16 == 0:
                result['likely_method'] = 'aes'
            elif len(data) % 64 == 0:
                result['likely_method'] = 'chacha20'
                
        return result
        
    def _get_entry_type(self, type_code: int) -> str:
        """Convert entry type code to string"""
        TYPE_CODES = {
            0x01: 'private',
            0x02: 'public',
            0x03: 'pool',
            0x04: 'master',
            0x05: 'settings'
        }
        return TYPE_CODES.get(type_code, 'unknown')
        
    def _identify_db_key_type(self, key: bytes) -> str:
        """Identify Berkeley DB key type"""
        if key.startswith(b'name'):
            return 'name'
        elif key.startswith(b'key'):
            return 'key'
        elif key.startswith(b'pool'):
            return 'pool'
        elif key.startswith(b'version'):
            return 'version'
        elif key.startswith(b'setting'):
            return 'setting'
        return 'unknown'

    def _check_legacy_encryption(self, header: bytes) -> bool:
        """Check for legacy encryption patterns"""
        try:
            # Check for common legacy patterns
            if len(header) < 8:
                return False
                
            # Check for old Bitcoin Core encryption
            if header.startswith(b'\x00\x00'):
                return True
                
            # Check for old MultiBit encryption
            if header.startswith(b'\x01\x00'):
                return True
                
            return False
            
        except Exception as e:
            self.logger.debug(f"Legacy encryption check failed: {e}")
            return False
            
    def _parse_db_flags(self, flag_bytes: bytes) -> List[str]:
        """Parse database entry flags
        
        Args:
            flag_bytes: Raw flag bytes
            
        Returns:
            List of flag strings
        """
        flags = []
        try:
            if len(flag_bytes) >= 4:
                flag_value = int.from_bytes(flag_bytes, byteorder='little')
                
                if flag_value & 0x01:
                    flags.append('encrypted')
                if flag_value & 0x02:
                    flags.append('compressed')
                if flag_value & 0x04:
                    flags.append('private')
                if flag_value & 0x08:
                    flags.append('pool')
                    
        except Exception as e:
            self.logger.debug(f"Flag parsing failed: {e}")
            
        return flags
        
    def _parse_timestamp(self, timestamp_bytes: bytes) -> Optional[datetime]:
        """Parse timestamp with validation
        
        Args:
            timestamp_bytes: Raw timestamp bytes
            
        Returns:
            Datetime object or None if invalid
        """
        try:
            if len(timestamp_bytes) < 8:
                return None
                
            timestamp = int.from_bytes(timestamp_bytes, byteorder='little')
            
            # Validate timestamp range
            if timestamp < 0:
                return None
                
            # Convert to datetime with validation
            dt = datetime.fromtimestamp(timestamp)
            
            # Check if date is reasonable
            if dt.year < MINYEAR or dt.year > MAXYEAR:
                return None
                
            return dt
            
        except (ValueError, OSError, OverflowError) as e:
            self.logger.debug(f"Timestamp parsing failed: {e}")
            return None
            
    def _validate_key_entry(self, entry: Dict[str, any]) -> bool:
        """Validate parsed key entry
        
        Args:
            entry: Parsed key entry
            
        Returns:
            True if entry is valid
        """
        try:
            # Check required fields
            if not entry.get('type'):
                return False
                
            # Validate key data if present
            if entry.get('key'):
                if not isinstance(entry['key'], (bytes, str)):
                    return False
                    
            # Validate timestamp if present
            if entry.get('timestamp'):
                if not isinstance(entry['timestamp'], datetime):
                    return False
                    
            return True
            
        except Exception:
            return False

    def _extract_compatibility_features(self, header: Dict[str, any]) -> List[float]:
        """Extract compatibility-related features
        
        Args:
            header: Wallet header dictionary
            
        Returns:
            List of compatibility features
        """
        features = []
        
        # Minimum version compatibility
        min_version = header.get('min_version', 0)
        features.append(float(min_version) / 100.0)
        
        # Maximum version compatibility
        max_version = header.get('max_version', 999999)
        features.append(float(max_version) / 100.0)
        
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
        
        # Feature compatibility flags
        compat_flags = header.get('compatibility_flags', [])
        flag_features = [
            1.0 if 'segwit' in compat_flags else 0.0,
            1.0 if 'taproot' in compat_flags else 0.0,
            1.0 if 'script_v2' in compat_flags else 0.0,
            1.0 if 'descriptor' in compat_flags else 0.0
        ]
        features.extend(flag_features)
        
        return features

    def _check_custom_patterns(self, data: bytes) -> bool:
        """Check for custom encryption patterns
        
        Args:
            data: Raw wallet data bytes
            
        Returns:
            True if custom pattern detected
        """
        try:
            # Common encryption patterns
            patterns = {
                'pkcs8': b'\x01\x30\x82',
                'asn1': b'\x02\x01\x00',
                'der': b'\x30\x82\x01',
                'pem': b'\x4D\x49\x47',
                'bdb': b'\x00\x05\x31',
                'sqlite': b'SQLite format 3',
                'berkeley': b'BerkeleyDB'
            }
            
            # Check first 1024 bytes for patterns
            header = data[:1024]
            
            # Look for known patterns
            for pattern in patterns.values():
                if pattern in header:
                    self.logger.debug(f"Found custom pattern: {pattern}")
                    return True
                    
            # Check for repeated patterns
            for i in range(0, len(header)-8, 8):
                block = header[i:i+8]
                if header.count(block) > 3:  # Repeated pattern detected
                    self.logger.debug("Found repeated pattern")
                    return True
                    
            # Check for high entropy blocks
            if self._check_high_entropy(header):
                self.logger.debug("Found high entropy block")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking custom patterns: {str(e)}")
            return False
            
    def _validate_db_entry(self, entry: Dict[str, any]) -> bool:
        """Validate database entry structure and content
        
        Args:
            entry: Database entry dictionary
            
        Returns:
            True if entry is valid
        """
        try:
            # Check if entry is a dictionary
            if not isinstance(entry, dict):
                return False
                
            # Required fields
            required_fields = ['type', 'key', 'value', 'flags']
            if not all(field in entry for field in required_fields):
                return False
                
            # Validate type
            valid_types = ['key', 'tx', 'addr', 'meta', 'setting', 'script']
            if entry['type'] not in valid_types:
                return False
                
            # Validate key
            if not isinstance(entry['key'], (bytes, str)):
                return False
                
            # Validate value
            if entry['value'] is None:
                return False
                
            # Type-specific validation
            if entry['type'] == 'key':
                if not self._validate_key_entry(entry):
                    return False
            elif entry['type'] == 'tx':
                if not self._validate_tx_entry(entry):
                    return False
            elif entry['type'] == 'addr':
                if not self._validate_addr_entry(entry):
                    return False
                    
            # Validate flags
            if not isinstance(entry['flags'], (list, tuple)):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating DB entry: {str(e)}")
            return False
            
    def _validate_key_entry(self, entry: Dict[str, any]) -> bool:
        """Validate key-type entry
        
        Args:
            entry: Key entry dictionary
            
        Returns:
            True if entry is valid
        """
        try:
            # Check key data
            key_data = entry.get('value', {})
            if not isinstance(key_data, dict):
                return False
                
            # Required key fields
            required_key_fields = ['key_type', 'key_data']
            if not all(field in key_data for field in required_key_fields):
                return False
                
            # Validate key type
            valid_key_types = ['private', 'public', 'master', 'derived']
            if key_data['key_type'] not in valid_key_types:
                return False
                
            # Validate key data
            if not isinstance(key_data['key_data'], bytes):
                return False
                
            return True
            
        except Exception as e:
            self.logger.debug(f"Key entry validation failed: {str(e)}")
            return False
            
    def _validate_tx_entry(self, entry: Dict[str, any]) -> bool:
        """Validate transaction-type entry
        
        Args:
            entry: Transaction entry dictionary
            
        Returns:
            True if entry is valid
        """
        try:
            # Check transaction data
            tx_data = entry.get('value', {})
            if not isinstance(tx_data, dict):
                return False
                
            # Required transaction fields
            required_tx_fields = ['txid', 'version', 'inputs', 'outputs']
            if not all(field in tx_data for field in required_tx_fields):
                return False
                
            # Validate txid
            if not isinstance(tx_data['txid'], (bytes, str)):
                return False
                
            # Validate version
            if not isinstance(tx_data['version'], int):
                return False
                
            # Validate inputs/outputs
            if not isinstance(tx_data['inputs'], list) or not isinstance(tx_data['outputs'], list):
                return False
                
            return True
            
        except Exception as e:
            self.logger.debug(f"Transaction entry validation failed: {str(e)}")
            return False
            
    def _validate_addr_entry(self, entry: Dict[str, any]) -> bool:
        """Validate address-type entry
        
        Args:
            entry: Address entry dictionary
            
        Returns:
            True if entry is valid
        """
        try:
            # Check address data
            addr_data = entry.get('value', {})
            if not isinstance(addr_data, dict):
                return False
                
            # Required address fields
            required_addr_fields = ['address', 'type', 'script']
            if not all(field in addr_data for field in required_addr_fields):
                return False
                
            # Validate address string
            if not isinstance(addr_data['address'], str):
                return False
                
            # Validate address type
            valid_addr_types = ['p2pkh', 'p2sh', 'p2wpkh', 'p2wsh', 'p2tr']
            if addr_data['type'] not in valid_addr_types:
                return False
                
            # Validate script
            if not isinstance(addr_data['script'], bytes):
                return False
                
            return True
            
        except Exception as e:
            self.logger.debug(f"Address entry validation failed: {str(e)}")
            return False
            
    def _check_high_entropy(self, data: bytes, threshold: float = 0.8) -> bool:
        """Check if data block has high entropy
        
        Args:
            data: Data bytes to check
            threshold: Entropy threshold (0-1)
            
        Returns:
            True if high entropy detected
        """
        try:
            if len(data) < 32:
                return False
                
            # Calculate byte frequency
            freq = {}
            for byte in data:
                freq[byte] = freq.get(byte, 0) + 1
                
            # Calculate entropy
            entropy = 0
            for count in freq.values():
                p = count / len(data)
                entropy -= p * math.log2(p)
                
            # Normalize entropy (max is 8 bits)
            normalized_entropy = entropy / 8.0
            
            return normalized_entropy > threshold
            
        except Exception as e:
            self.logger.debug(f"Entropy calculation failed: {str(e)}")
            return False

    def _calculate_entropy(self, data: bytes, block_size: int = 256) -> float:
        """Calculate Shannon entropy of data
        
        Args:
            data: Bytes to analyze
            block_size: Size of data block to analyze
            
        Returns:
            Entropy value between 0 and 1
        """
        try:
            # Use a sample if data is too large
            if len(data) > block_size:
                data = data[:block_size]
                
            # Calculate byte frequency
            freq = {}
            for byte in data:
                freq[byte] = freq.get(byte, 0) + 1
                
            # Calculate entropy
            entropy = 0
            for count in freq.values():
                p = count / len(data)
                entropy -= p * math.log2(p)
                
            # Normalize entropy (max is 8 bits)
            return entropy / 8.0
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0