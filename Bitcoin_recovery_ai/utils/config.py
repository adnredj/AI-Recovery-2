import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
from copy import deepcopy

from .logging import setup_logger

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If configuration loading fails
    """
    logger = setup_logger(__name__)
    
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
            
        # Load configuration based on file extension
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config = _load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            config = _load_json(config_path)
        else:
            raise ConfigError(f"Unsupported configuration format: {config_path.suffix}")
            
        # Validate and process configuration
        config = _process_config(config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        raise ConfigError(f"Error loading configuration: {str(e)}")

def _load_yaml(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Error loading YAML configuration: {str(e)}")

def _load_json(config_path: Path) -> Dict[str, Any]:
    """Load JSON configuration file
    
    Args:
        config_path: Path to JSON file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ConfigError(f"Error loading JSON configuration: {str(e)}")

def _process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process and validate configuration
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Processed configuration dictionary
    """
    # Create a copy to avoid modifying original
    config = deepcopy(config)
    
    # Add default values
    config = _add_defaults(config)
    
    # Validate required fields
    _validate_config(config)
    
    # Process paths
    config = _process_paths(config)
    
    # Process GPU settings
    config = _process_gpu_config(config)
    
    return config

def _add_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add default values to configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with defaults added
    """
    defaults = {
        'model': {
            'type': 'transformer',
            'embedding_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 100,
            'early_stopping_patience': 10,
            'gradient_clip_val': 1.0
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'num_workers': 4
        },
        'logging': {
            'level': 'INFO',
            'save_dir': 'logs'
        },
        'gpu': {
            'use_gpu': True,
            'gpu_id': 0,
            'memory_fraction': 0.8
        }
    }
    
    # Update configuration with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(default_value, dict):
            for subkey, subvalue in default_value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue
                    
    return config

def _validate_config(config: Dict[str, Any]):
    """Validate configuration requirements
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigError: If validation fails
    """
    required_fields = [
        'model',
        'training',
        'data'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ConfigError(f"Missing required configuration field: {field}")
            
    # Validate model configuration
    if 'model' in config:
        if 'type' not in config['model']:
            raise ConfigError("Model type not specified in configuration")
            
    # Validate training configuration
    if 'training' in config:
        if 'batch_size' not in config['training']:
            raise ConfigError("Batch size not specified in training configuration")
            
    # Validate data configuration
    if 'data' in config:
        splits = config['data'].get('train_split', 0) + \
                config['data'].get('val_split', 0) + \
                config['data'].get('test_split', 0)
        if not (0.99 <= splits <= 1.01):  # Allow small floating point errors
            raise ConfigError("Data splits must sum to 1.0")

def _process_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process and validate paths in configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with processed paths
    """
    # Process data paths
    if 'data' in config:
        if 'data_dir' in config['data']:
            config['data']['data_dir'] = str(Path(config['data']['data_dir']).resolve())
            
    # Process output paths
    if 'output_dir' in config:
        config['output_dir'] = str(Path(config['output_dir']).resolve())
        
    # Process checkpoint paths
    if 'checkpoint_dir' in config:
        config['checkpoint_dir'] = str(Path(config['checkpoint_dir']).resolve())
        
    return config

def _process_gpu_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process GPU configuration settings
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with processed GPU settings
    """
    if 'gpu' in config:
        # Validate memory fraction
        if 'memory_fraction' in config['gpu']:
            fraction = config['gpu']['memory_fraction']
            if not (0 < fraction <= 1):
                config['gpu']['memory_fraction'] = 0.8
                
        # Validate GPU ID
        if 'gpu_id' in config['gpu']:
            if config['gpu']['gpu_id'] < 0:
                config['gpu']['gpu_id'] = 0
                
    return config

# Export functions
__all__ = ['load_config', 'ConfigError']