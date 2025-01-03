from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import hashlib
import bitcoin
from ..utils.logging import setup_logger
from ..crypto.recovery import BitcoinRecoveryEngine
from ..analysis.pattern_analyzer import PatternAnalyzer
from ..validation.transaction_validator import TransactionValidator

class TaskExecutor:
    """Handles execution of specific task types"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.recovery_engine = BitcoinRecoveryEngine(config)
        self.pattern_analyzer = PatternAnalyzer(config)
        self.validator = TransactionValidator(config)
        
    def execute_recovery_task(self, task: Dict[str, any]) -> Dict[str, any]:
        """Execute bitcoin wallet recovery task"""
        self.logger.info(f"Starting recovery task {task['task_id']}")
        
        # Extract task parameters
        wallet_data = task['wallet_data']
        recovery_params = task['recovery_params']
        search_space = task.get('search_space', {})
        
        # Validate inputs
        self._validate_recovery_inputs(wallet_data, recovery_params)
        
        # Initialize recovery context
        context = self._initialize_recovery_context(wallet_data)
        
        # Execute recovery process
        try:
            result = self.recovery_engine.recover(
                context=context,
                params=recovery_params,
                search_space=search_space,
                progress_callback=self._update_progress
            )
            
            # Validate results
            if result.get('success'):
                validation = self._validate_recovery_result(result, wallet_data)
                result['validation'] = validation
                
            return {
                'status': 'completed',
                'result': result,
                'metrics': self._get_recovery_metrics(context)
            }
            
        except Exception as e:
            self.logger.error(f"Recovery task failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'metrics': self._get_recovery_metrics(context)
            }
    
    def execute_analysis_task(self, task: Dict[str, any]) -> Dict[str, any]:
        """Execute pattern analysis task"""
        self.logger.info(f"Starting analysis task {task['task_id']}")
        
        # Extract task parameters
        data = task['data']
        analysis_type = task['analysis_type']
        analysis_params = task.get('params', {})
        
        try:
            # Perform analysis based on type
            if analysis_type == 'pattern_recognition':
                result = self.pattern_analyzer.analyze_patterns(data, analysis_params)
            elif analysis_type == 'entropy_analysis':
                result = self.pattern_analyzer.analyze_entropy(data, analysis_params)
            elif analysis_type == 'statistical_analysis':
                result = self.pattern_analyzer.analyze_statistics(data, analysis_params)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
            return {
                'status': 'completed',
                'result': result,
                'metrics': self._get_analysis_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Analysis task failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'metrics': self._get_analysis_metrics()
            }
    
    def execute_validation_task(self, task: Dict[str, any]) -> Dict[str, any]:
        """Execute validation task"""
        self.logger.info(f"Starting validation task {task['task_id']}")
        
        # Extract task parameters
        data = task['data']
        validation_type = task['validation_type']
        validation_params = task.get('params', {})
        
        try:
            # Perform validation based on type
            if validation_type == 'transaction':
                result = self.validator.validate_transaction(data, validation_params)
            elif validation_type == 'address':
                result = self.validator.validate_address(data, validation_params)
            elif validation_type == 'signature':
                result = self.validator.validate_signature(data, validation_params)
            else:
                raise ValueError(f"Unknown validation type: {validation_type}")
                
            return {
                'status': 'completed',
                'result': result,
                'metrics': self._get_validation_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Validation task failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'metrics': self._get_validation_metrics()
            }
    
    def _validate_recovery_inputs(self, 
                                wallet_data: Dict[str, any],
                                recovery_params: Dict[str, any]):
        """Validate recovery task inputs"""
        required_wallet_fields = ['type', 'data', 'version']
        required_params_fields = ['method', 'depth', 'timeout']
        
        for field in required_wallet_fields:
            if field not in wallet_data:
                raise ValueError(f"Missing required wallet field: {field}")
                
        for field in required_params_fields:
            if field not in recovery_params:
                raise ValueError(f"Missing required parameter: {field}")
                
        # Validate wallet type
        if wallet_data['type'] not in self.config.supported_wallet_types:
            raise ValueError(f"Unsupported wallet type: {wallet_data['type']}")
            
        # Validate recovery method
        if recovery_params['method'] not in self.config.supported_recovery_methods:
            raise ValueError(f"Unsupported recovery method: {recovery_params['method']}")
    
    def _initialize_recovery_context(self, 
                                   wallet_data: Dict[str, any]) -> Dict[str, any]:
        """Initialize recovery context"""
        return {
            'start_time': datetime.now(),
            'wallet_type': wallet_data['type'],
            'search_stats': {
                'combinations_tested': 0,
                'patterns_matched': 0,
                'current_depth': 0
            },
            'checkpoints': [],
            'progress': 0.0
        }
    
    def _validate_recovery_result(self,
                                result: Dict[str, any],
                                wallet_data: Dict[str, any]) -> Dict[str, bool]:
        """Validate recovery results"""
        if not result.get('recovered_key'):
            return {'valid': False, 'reason': 'No key recovered'}
            
        try:
            # Verify recovered key matches wallet data
            key_valid = self.validator.validate_key(
                result['recovered_key'],
                wallet_data
            )
            
            if not key_valid:
                return {'valid': False, 'reason': 'Invalid key'}
                
            # Verify key can generate correct addresses
            address_valid = self.validator.validate_addresses(
                result['recovered_key'],
                wallet_data.get('addresses', [])
            )
            
            if not address_valid:
                return {'valid': False, 'reason': 'Address mismatch'}
                
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'reason': str(e)}
    
    def _update_progress(self, context: Dict[str, any], progress: float):
        """Update recovery progress"""
        context['progress'] = progress
        
        # Create checkpoint if needed
        if (progress - context.get('last_checkpoint_progress', 0)) >= self.config.checkpoint_interval:
            self._create_checkpoint(context)
            context['last_checkpoint_progress'] = progress
    
    def _create_checkpoint(self, context: Dict[str, any]):
        """Create recovery checkpoint"""
        checkpoint = {
            'timestamp': datetime.now(),
            'progress': context['progress'],
            'stats': context['search_stats'].copy(),
            'state': self.recovery_engine.get_state()
        }
        context['checkpoints'].append(checkpoint)
        
    def _get_recovery_metrics(self, context: Dict[str, any]) -> Dict[str, any]:
        """Get recovery process metrics"""
        end_time = datetime.now()
        duration = (end_time - context['start_time']).total_seconds()
        
        return {
            'duration': duration,
            'combinations_tested': context['search_stats']['combinations_tested'],
            'patterns_matched': context['search_stats']['patterns_matched'],
            'progress': context['progress'],
            'checkpoints_created': len(context['checkpoints']),
            'average_speed': context['search_stats']['combinations_tested'] / duration
                if duration > 0 else 0
        }