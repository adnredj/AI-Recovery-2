#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple
import argparse
import sys
import yaml
from pathlib import Path
import logging
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader

from bitcoin_recovery_ai.models.architecture.encoder import WalletEncoder
from bitcoin_recovery_ai.models.architecture.decoder import WalletDecoder
from bitcoin_recovery_ai.models.architecture.attention import SelfAttention
from bitcoin_recovery_ai.training.trainer import ModelTrainer
from bitcoin_recovery_ai.training.loss_functions import RecoveryLoss
from bitcoin_recovery_ai.training.metrics import RecoveryMetrics
from bitcoin_recovery_ai.preprocessing.wallet_parser import WalletParser
from bitcoin_recovery_ai.preprocessing.feature_extractor import FeatureExtractor
from bitcoin_recovery_ai.preprocessing.data_cleaner import DataCleaner
from bitcoin_recovery_ai.validation.validator import RecoveryValidator
from bitcoin_recovery_ai.utils.gpu_utils import setup_gpu
from bitcoin_recovery_ai.utils.metrics import calculate_metrics
from bitcoin_recovery_ai.utils.config import load_config, ConfigError
from bitcoin_recovery_ai.utils.logging import setup_logger

class BitcoinRecoveryAI:
    """Autonomous AI system for Bitcoin wallet recovery"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(__name__, self.config.get('logging', {}))
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize AI components"""
        try:
            # Setup GPU
            self.device = setup_gpu(self.config.get('gpu', {}))
            
            # Initialize neural network components
            self.encoder = WalletEncoder(self.config['model']['encoder']).to(self.device)
            self.decoder = WalletDecoder(self.config['model']['decoder']).to(self.device)
            self.attention = SelfAttention(self.config['model']['attention']).to(self.device)
            
            # Initialize preprocessing
            self.wallet_parser = WalletParser(self.config)
            self.feature_extractor = FeatureExtractor(self.config)
            self.data_cleaner = DataCleaner(self.config)
            
            # Initialize training components
            self.loss_fn = RecoveryLoss(self.config)
            self.metrics = RecoveryMetrics(self.config)
            self.trainer = ModelTrainer(
                encoder=self.encoder,
                decoder=self.decoder,
                attention=self.attention,
                loss_fn=self.loss_fn,
                metrics=self.metrics,
                config=self.config
            )
            
            # Initialize validator
            self.validator = RecoveryValidator(self.config)
            
            # Load pretrained model if available
            self._load_pretrained_model()
            
            self.logger.info("AI components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing AI components: {str(e)}")
            raise
            
    def train(self, 
             training_data_path: str,
             validation_data_path: Optional[str] = None,
             **kwargs):
        """Train the AI model"""
        self.logger.info("Starting autonomous training process")
        
        try:
            # Load and preprocess training data
            train_data = self._prepare_training_data(training_data_path)
            val_data = self._prepare_training_data(validation_data_path) if validation_data_path else None
            
            # Create data loaders
            train_loader = DataLoader(
                train_data,
                batch_size=self.config['training']['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(val_data, batch_size=self.config['training']['batch_size']) if val_data else None
            
            # Start training
            training_result = self.trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                **kwargs
            )
            
            # Evaluate and save model
            self._evaluate_and_save_model(training_result)
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
    def autonomous_improve(self, feedback_data: Dict[str, any]):
        """Autonomous model improvement based on feedback"""
        self.logger.info("Starting autonomous improvement process")
        
        try:
            # Analyze feedback
            improvement_strategy = self._analyze_feedback(feedback_data)
            
            # Adjust model architecture if needed
            if improvement_strategy.get('adjust_architecture'):
                self._adjust_model_architecture(improvement_strategy['architecture_changes'])
            
            # Fine-tune model
            if improvement_strategy.get('fine_tune'):
                self._fine_tune_model(feedback_data)
            
            # Update hyperparameters
            if improvement_strategy.get('adjust_hyperparams'):
                self._adjust_hyperparameters(improvement_strategy['hyperparameter_changes'])
            
            # Validate improvements
            validation_result = self._validate_improvements(feedback_data)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Autonomous improvement failed: {str(e)}")
            raise
            
    def recover_wallet(self, 
                      wallet_path: str,
                      output_path: Optional[str] = None,
                      **kwargs) -> Dict[str, any]:
        """Perform wallet recovery using trained AI model"""
        start_time = datetime.now()
        self.logger.info(f"Starting AI recovery process for wallet: {wallet_path}")
        
        try:
            # Prepare input data
            wallet_data = self.wallet_parser.parse_wallet(wallet_path)
            cleaned_data = self.data_cleaner.clean_wallet_data(wallet_data)
            features = self.feature_extractor.extract_features(cleaned_data)
            
            # Convert to tensor and move to device
            input_tensor = torch.tensor(features).to(self.device)
            
            # Generate embeddings
            embeddings = self.encoder(input_tensor)
            
            # Apply attention
            attended = self.attention(embeddings)
            
            # Generate recovery result
            recovery_result = self.decoder(attended)
            
            # Post-process results
            processed_result = self._post_process_results(recovery_result)
            
            # Validate results
            validation_result = self.validator.validate_recovery(
                processed_result,
                cleaned_data
            )
            
            # Calculate metrics
            metrics = self.metrics.calculate_metrics(
                processed_result,
                cleaned_data,
                validation_result
            )
            
            # Prepare output
            result = {
                'status': 'success',
                'recovery_result': processed_result,
                'validation': validation_result,
                'metrics': metrics,
                'confidence_score': self._calculate_confidence(processed_result),
                'duration': (datetime.now() - start_time).total_seconds()
            }
            
            # Save results if needed
            if output_path:
                self._save_results(result, output_path)
                
            # Update model with new experience
            self._update_model_experience(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"AI recovery failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds()
            }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bitcoin Wallet Recovery AI')
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'recover'],
        required=True,
        help='Operation mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--wallet',
        type=str,
        help='Path to wallet file (for recover mode)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (for evaluate/recover modes)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Setup logging
        logger = setup_logger(__name__)
        
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
            
        config = load_config(config_path)
        
        # Update config with command line arguments
        if args.output:
            config['output_dir'] = args.output
        if args.checkpoint:
            config['checkpoint_path'] = args.checkpoint
            
        # Setup GPU if available
        device = setup_gpu(config)
        
        # Execute requested mode
        if args.mode == 'train':
            train(config, device)
        elif args.mode == 'evaluate':
            evaluate(config, device)
        elif args.mode == 'recover':
            if not args.wallet:
                raise ValueError("Wallet path required for recover mode")
            recover(args.wallet, config, device)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

def train(config: Dict, device: str):
    """Train the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting training mode")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config=config, device=device)
        
        # Create data loaders
        train_loader = create_train_loader(config)
        val_loader = create_val_loader(config)
        
        # Train model
        trainer.train(train_loader, val_loader)
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise

def evaluate(config: Dict, device: str):
    """Evaluate the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation mode")
    
    try:
        validator = RecoveryValidator(config, device)
        results = validator.evaluate()
        logger.info(f"Evaluation results: {results}")
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise

def recover(wallet_path: str, config: Dict, device: str):
    """Recover wallet"""
    logger = logging.getLogger(__name__)
    logger.info("Starting recovery mode")
    
    try:
        # Parse wallet
        parser = WalletParser(config)
        wallet_data = parser.parse_wallet(wallet_path)
        
        # Extract features
        extractor = FeatureExtractor(config)
        features = extractor.extract_features(wallet_data)
        
        # Run recovery
        validator = RecoveryValidator(config, device)
        result = validator.recover(features)
        
        logger.info(f"Recovery result: {result}")
    except Exception as e:
        logger.error(f"Recovery error: {str(e)}")
        raise

if __name__ == '__main__':
    main()