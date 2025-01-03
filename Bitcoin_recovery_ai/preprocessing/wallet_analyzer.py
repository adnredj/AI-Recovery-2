import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ..utils.crypto_utils import CryptoUtils
from ..utils.logging import setup_logger
from ..utils.db_utils import BerkeleyDBUtils

class WalletAnalyzer:
    """Advanced wallet format detection and analysis"""
    
    def __init__(self, config):
        self.config = config
        self.crypto_utils = CryptoUtils()
        self.db_utils = BerkeleyDBUtils()
        self.logger = setup_logger(__name__)
        
    def analyze_wallet(self, wallet_path: Path) -> Dict[str, any]:
        """Perform comprehensive wallet analysis"""
        with open(wallet_path, 'rb') as f:
            wallet_data = f.read()
            
        analysis = {
            'format': self._detect_wallet_format(wallet_data),
            'structure': self._analyze_wallet_structure(wallet_data),
            'metadata': self._extract_wallet_metadata(wallet_data),
            'key_info': self._analyze_key_information(wallet_data),
            'encryption_info': self._analyze_encryption_info(wallet_data),
            'database_info': self._analyze_database_structure(wallet_data),
            'recovery_recommendations': []
        }
        
        # Generate recovery recommendations
        analysis['recovery_recommendations'] = self._generate_recommendations(analysis)
        return analysis
    
    def _detect_wallet_format(self, data: bytes) -> Dict[str, any]:
        """Detect wallet format and version"""
        format_info = {
            'type': None,
            'version': None,
            'confidence': 0.0,
            'indicators': []
        }
        
        # Check for Berkeley DB format
        if self._check_berkeley_db_format(data):
            format_info.update({
                'type': 'berkeley_db',
                'version': self._detect_db_version(data),
                'confidence': self._calculate_db_confidence(data)
            })
            
        # Check for Bitcoin Core format
        elif self._check_bitcoin_core_format(data):
            format_info.update({
                'type': 'bitcoin_core',
                'version': self._detect_core_version(data),
                'confidence': self._calculate_core_confidence(data)
            })
            
        # Add format-specific indicators
        format_info['indicators'] = self._collect_format_indicators(data, format_info['type'])
        return format_info
    
    def _analyze_wallet_structure(self, data: bytes) -> Dict[str, any]:
        """Analyze wallet file structure"""
        return {
            'file_structure': self._analyze_file_structure(data),
            'key_storage': self._analyze_key_storage(data),
            'header_info': self._analyze_header(data),
            'sections': self._identify_sections(data)
        }
    
    def _extract_wallet_metadata(self, data: bytes) -> Dict[str, any]:
        """Extract wallet metadata"""
        metadata = {
            'creation_time': self._extract_creation_time(data),
            'last_update': self._extract_last_update(data),
            'key_pool_size': self._extract_key_pool_size(data),
            'version_info': self._extract_version_info(data),
            'network_type': self._detect_network_type(data),
            'features': self._detect_wallet_features(data)
        }
        
        # Add format-specific metadata
        if metadata.get('version_info', {}).get('type') == 'bitcoin_core':
            metadata.update(self._extract_core_specific_metadata(data))
            
        return metadata
    
    def _analyze_key_information(self, data: bytes) -> Dict[str, any]:
        """Analyze key-related information"""
        return {
            'key_types': self._identify_key_types(data),
            'key_derivation': self._analyze_key_derivation(data),
            'key_encryption': self._analyze_key_encryption(data),
            'master_key_info': self._analyze_master_key(data),
            'key_hierarchy': self._analyze_key_hierarchy(data)
        }
    
    def _analyze_encryption_info(self, data: bytes) -> Dict[str, any]:
        """Analyze encryption-related information"""
        encryption_info = {
            'method': self._detect_encryption_method(data),
            'parameters': self._extract_encryption_params(data),
            'key_derivation': self._analyze_encryption_key_derivation(data),
            'salt_info': self._analyze_salt_information(data)
        }
        
        # Add method-specific analysis
        if encryption_info['method'].get('type') == 'aes':
            encryption_info.update(self._analyze_aes_specifics(data))
            
        return encryption_info
    
    def _analyze_database_structure(self, data: bytes) -> Dict[str, any]:
        """Analyze database structure if applicable"""
        if not self._is_database_format(data):
            return {'is_database': False}
            
        return {
            'is_database': True,
            'db_type': self._detect_db_type(data),
            'db_version': self._detect_db_version(data),
            'table_structure': self._analyze_table_structure(data),
            'indexes': self._analyze_db_indexes(data),
            'metadata': self._extract_db_metadata(data)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate recovery recommendations based on analysis"""
        recommendations = []
        
        # Format-specific recommendations
        if analysis['format']['type'] == 'berkeley_db':
            recommendations.extend(self._generate_db_recommendations(analysis))
        elif analysis['format']['type'] == 'bitcoin_core':
            recommendations.extend(self._generate_core_recommendations(analysis))
            
        # Encryption-specific recommendations
        if analysis['encryption_info']['method']:
            recommendations.extend(self._generate_encryption_recommendations(analysis))
            
        # Key-specific recommendations
        recommendations.extend(self._generate_key_recommendations(analysis))
        
        # Sort recommendations by priority
        recommendations.sort(key=lambda x: x.get('priority', 0), reverse=True)
        return recommendations
    
    def _check_berkeley_db_format(self, data: bytes) -> bool:
        """Check if wallet is in Berkeley DB format"""
        try:
            # Check Berkeley DB magic number
            if not data.startswith(b'BDB'):
                return False
                
            # Check version header
            version_bytes = data[3:7]
            if not (version_bytes[0] >= 4 and version_bytes[0] <= 9):
                return False
                
            # Check page size (typically 4096 or 8192)
            page_size = struct.unpack('>I', data[12:16])[0]
            if page_size not in [4096, 8192]:
                return False
                
            # Check database flags
            flags = struct.unpack('>I', data[16:20])[0]
            if not (flags & 0x00000002):  # Check for DB_BTREE flag
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking Berkeley DB format: {str(e)}")
            return False
    
    def _check_bitcoin_core_format(self, data: bytes) -> bool:
        """Check if wallet is in Bitcoin Core format"""
        try:
            # Check magic bytes
            if not data.startswith(b'\xf9\xbe\xb4\xd9'):
                return False
                
            # Check wallet header
            if len(data) < 256:  # Minimum wallet size
                return False
                
            # Check format version
            version = struct.unpack('<I', data[4:8])[0]
            if version not in [40000, 60000, 70000]:  # Known versions
                return False
                
            # Check key pool
            if b'key pool' not in data[:1024]:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking Bitcoin Core format: {str(e)}")
            return False
    
    def _analyze_file_structure(self, data: bytes) -> Dict[str, any]:
        """Analyze file structure details"""
        try:
            structure = {
                'size': len(data),
                'sections': {},
                'offsets': {},
                'format_version': None,
                'created_at': None,
                'last_modified': None
            }
            
            # Analyze file sections
            sections = self._identify_sections(data)
            structure['sections'] = sections
            
            # Record important offsets
            structure['offsets'] = self._find_key_offsets(data)
            
            # Extract version
            structure['format_version'] = self._extract_version(data)
            
            # Extract timestamps
            timestamps = self._extract_timestamps(data)
            if timestamps:
                structure['created_at'] = timestamps.get('created_at')
                structure['last_modified'] = timestamps.get('last_modified')
                
            return structure
            
        except Exception as e:
            self.logger.error(f"Error analyzing file structure: {str(e)}")
            return {}
    
    def _analyze_key_storage(self, data: bytes) -> Dict[str, any]:
        """Analyze key storage mechanism"""
        try:
            storage = {
                'type': None,
                'encryption': None,
                'key_count': 0,
                'key_locations': [],
                'derivation_info': None,
                'hierarchy': None
            }
            
            # Detect storage type
            storage['type'] = self._detect_storage_type(data)
            
            # Analyze encryption
            storage['encryption'] = self._analyze_encryption_method(data)
            
            # Count and locate keys
            key_info = self._locate_keys(data)
            storage['key_count'] = key_info['count']
            storage['key_locations'] = key_info['locations']
            
            # Analyze key derivation
            storage['derivation_info'] = self._analyze_derivation(data)
            
            # Analyze key hierarchy
            storage['hierarchy'] = self._analyze_hierarchy(data)
            
            return storage
            
        except Exception as e:
            self.logger.error(f"Error analyzing key storage: {str(e)}")
            return {}
    
    def _detect_network_type(self, data: bytes) -> str:
        """Detect Bitcoin network type"""
        try:
            # Check network magic bytes
            if b'\xf9\xbe\xb4\xd9' in data[:1024]:
                return 'mainnet'
                
            if b'\x0b\x11\x09\x07' in data[:1024]:
                return 'testnet'
                
            if b'\xfa\xbf\xb5\xda' in data[:1024]:
                return 'regtest'
                
            # Check address prefixes
            if b'\x00' in data[:1024]:  # Mainnet addresses
                return 'mainnet'
                
            if b'\x6f' in data[:1024]:  # Testnet addresses
                return 'testnet'
                
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Error detecting network type: {str(e)}")
            return 'error'
    
    def _analyze_master_key(self, data: bytes) -> Dict[str, any]:
        """Analyze master key information"""
        try:
            master_key = {
                'present': False,
                'type': None,
                'encryption': None,
                'derivation_path': None,
                'fingerprint': None,
                'metadata': {}
            }
            
            # Look for master key markers
            if not self._find_master_key_section(data):
                return master_key
                
            master_key['present'] = True
            
            # Determine key type
            master_key['type'] = self._determine_master_key_type(data)
            
            # Analyze encryption
            master_key['encryption'] = self._analyze_master_key_encryption(data)
            
            # Extract derivation path
            master_key['derivation_path'] = self._extract_derivation_path(data)
            
            # Calculate fingerprint
            master_key['fingerprint'] = self._calculate_master_key_fingerprint(data)
            
            # Extract metadata
            master_key['metadata'] = self._extract_master_key_metadata(data)
            
            return master_key
            
        except Exception as e:
            self.logger.error(f"Error analyzing master key: {str(e)}")
            return {'present': False}

    def _identify_sections(self, data: bytes) -> Dict[str, any]:
        """Identify wallet file sections"""
        sections = {}
        current_pos = 0
        
        while current_pos < len(data):
            section = self._parse_section(data[current_pos:])
            if not section:
                break
            sections[section['type']] = section
            current_pos += section['size']
            
        return sections

    def _find_key_offsets(self, data: bytes) -> Dict[str, int]:
        """Find important key offsets in wallet file"""
        offsets = {}
        
        # Search for key markers
        markers = {
            'master_key': b'mkey',
            'key_pool': b'pool',
            'key_meta': b'keymeta',
            'encrypted_keys': b'ckey'
        }
        
        for name, marker in markers.items():
            pos = data.find(marker)
            if pos != -1:
                offsets[name] = pos
                
        return offsets

    def _detect_storage_type(self, data: bytes) -> str:
        """Detect key storage type"""
        if self._check_berkeley_db_format(data):
            return 'berkeley_db'
        elif self._check_bitcoin_core_format(data):
            return 'bitcoin_core'
        else:
            return 'unknown'

    def _analyze_hierarchy(self, data: bytes) -> Dict[str, any]:
        """Analyze key hierarchy structure"""
        hierarchy = {
            'type': None,
            'depth': 0,
            'branches': [],
            'is_hd': False
        }
        
        # Check for HD wallet markers
        if b'hd' in data[:1024]:
            hierarchy['is_hd'] = True
            hierarchy['type'] = 'bip32'
            
        # Analyze hierarchy depth
        hierarchy['depth'] = self._calculate_hierarchy_depth(data)
        
        # Identify branches
        hierarchy['branches'] = self._identify_hierarchy_branches(data)
        
        return hierarchy