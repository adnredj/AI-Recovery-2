import bsddb3
import struct
from typing import Dict, List, Optional
from pathlib import Path

class BerkeleyDBUtils:
    """Berkeley DB utilities for wallet analysis"""
    
    @staticmethod
    def analyze_db_structure(wallet_path: str) -> Dict[str, any]:
        """Analyze Berkeley DB structure of wallet"""
        analysis = {
            'version': None,
            'key_types': {},
            'metadata': {},
            'structure_type': None
        }
        
        try:
            db = bsddb3.btopen(wallet_path, 'r')
            
            # Analyze DB version and type
            if b'version' in db:
                analysis['version'] = struct.unpack('<I', db[b'version'])[0]
                
            # Analyze key types and patterns
            for key in db.keys():
                key_type = BerkeleyDBUtils._identify_key_type(key)
                if key_type:
                    analysis['key_types'][key_type] = analysis['key_types'].get(key_type, 0) + 1
                    
            # Determine structure type
            analysis['structure_type'] = BerkeleyDBUtils._determine_structure_type(analysis['version'])
            
            db.close()
            
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis

    @staticmethod
    def extract_keys(wallet_path: str) -> List[Dict[str, any]]:
        """Extract key information from Berkeley DB"""
        keys = []
        try:
            db = bsddb3.btopen(wallet_path, 'r')
            
            for key, value in db.items():
                if key.startswith(b'key'):
                    key_info = BerkeleyDBUtils._parse_key_record(key, value)
                    if key_info:
                        keys.append(key_info)
                        
            db.close()
            
        except Exception as e:
            pass
            
        return keys

    @staticmethod
    def _identify_key_type(key: bytes) -> Optional[str]:
        """Identify type of DB key"""
        if key.startswith(b'key'):
            return 'private_key'
        elif key.startswith(b'name'):
            return 'name'
        elif key.startswith(b'tx'):
            return 'transaction'
        elif key.startswith(b'acc'):
            return 'account'
        return None

    @staticmethod
    def _determine_structure_type(version: Optional[int]) -> str:
        """Determine Berkeley DB structure type based on version"""
        if not version:
            return 'unknown'
        elif version < 40000:
            return 'early_bitcoin_core'
        elif version < 60000:
            return 'mid_bitcoin_core'
        else:
            return 'modern_bitcoin_core'

    @staticmethod
    def _parse_key_record(key: bytes, value: bytes) -> Optional[Dict[str, any]]:
        """Parse key record from Berkeley DB"""
        try:
            return {
                'key_type': 'private_key',
                'key_data': value[8:40] if len(value) >= 40 else value,
                'creation_time': struct.unpack('<I', value[0:4])[0] if len(value) >= 4 else None,
                'flags': value[4:8] if len(value) >= 8 else None
            }
        except Exception:
            return None