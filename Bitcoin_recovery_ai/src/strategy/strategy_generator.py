import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from ..utils.logging import setup_logger
from ..utils.crypto_utils import CryptoUtils
from ..analysis.pattern_analyzer import PatternAnalyzer
from ..analysis.wallet_analyzer import WalletAnalyzer

class RecoveryStrategyGenerator:
    """Generate optimized recovery strategies based on wallet analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.crypto_utils = CryptoUtils()
        self.pattern_analyzer = PatternAnalyzer(config)
        self.wallet_analyzer = WalletAnalyzer(config)
        
    def generate_strategies(self, 
                          wallet_path: Path,
                          analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Generate comprehensive recovery strategies"""
        
        # Generate different types of strategies
        strategies = {
            'primary_strategies': self._generate_primary_strategies(analysis_results),
            'fallback_strategies': self._generate_fallback_strategies(analysis_results),
            'optimization_strategies': self._generate_optimization_strategies(analysis_results),
            'parallel_strategies': self._generate_parallel_strategies(analysis_results)
        }
        
        # Rank and prioritize strategies
        ranked_strategies = self._rank_strategies(strategies)
        
        # Generate execution plan
        execution_plan = self._generate_execution_plan(ranked_strategies)
        
        return {
            'strategies': ranked_strategies,
            'execution_plan': execution_plan,
            'estimated_resources': self._estimate_required_resources(ranked_strategies),
            'success_probability': self._estimate_success_probability(ranked_strategies),
            'recommendations': self._generate_recommendations(ranked_strategies)
        }
    
    def _generate_primary_strategies(self, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate primary recovery strategies"""
        strategies = []
        
        # Pattern-based strategies
        if analysis.get('byte_patterns'):
            strategies.extend(self._generate_pattern_based_strategies(
                analysis['byte_patterns']
            ))
            
        # Structure-based strategies
        if analysis.get('structural_patterns'):
            strategies.extend(self._generate_structure_based_strategies(
                analysis['structural_patterns']
            ))
            
        # Encryption-based strategies
        if analysis.get('encryption_info'):
            strategies.extend(self._generate_encryption_based_strategies(
                analysis['encryption_info']
            ))
            
        return strategies
    
    def _generate_fallback_strategies(self, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate fallback recovery strategies"""
        strategies = []
        
        # Generate alternative approaches
        strategies.extend(self._generate_alternative_approaches(analysis))
        
        # Generate brute force strategies
        if self.config.allow_brute_force:
            strategies.extend(self._generate_brute_force_strategies(analysis))
            
        # Generate hybrid strategies
        strategies.extend(self._generate_hybrid_strategies(analysis))
        
        return strategies
    
    def _generate_optimization_strategies(self, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate optimization strategies"""
        strategies = []
        
        # Resource optimization
        strategies.extend(self._generate_resource_optimization_strategies(analysis))
        
        # Time optimization
        strategies.extend(self._generate_time_optimization_strategies(analysis))
        
        # Parallel processing optimization
        strategies.extend(self._generate_parallel_optimization_strategies(analysis))
        
        return strategies
    
    def _generate_parallel_strategies(self, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate parallel execution strategies"""
        return {
            'gpu_strategies': self._generate_gpu_strategies(analysis),
            'cpu_strategies': self._generate_cpu_strategies(analysis),
            'distributed_strategies': self._generate_distributed_strategies(analysis),
            'hybrid_execution': self._generate_hybrid_execution_strategies(analysis)
        }
    
    def _rank_strategies(self, strategies: Dict[str, List[Dict[str, any]]]) -> List[Dict[str, any]]:
        """Rank and prioritize recovery strategies"""
        all_strategies = []
        
        # Collect all strategies
        for strategy_type, strategy_list in strategies.items():
            for strategy in strategy_list:
                strategy['type'] = strategy_type
                all_strategies.append(strategy)
                
        # Calculate scores
        scored_strategies = [
            {
                **strategy,
                'score': self._calculate_strategy_score(strategy)
            }
            for strategy in all_strategies
        ]
        
        # Sort by score
        return sorted(scored_strategies, key=lambda x: x['score'], reverse=True)
    
    def _calculate_strategy_score(self, strategy: Dict[str, any]) -> float:
        """Calculate strategy score based on multiple factors"""
        score = 0.0
        
        # Success probability
        score += strategy.get('success_probability', 0) * self.config.probability_weight
        
        # Resource efficiency
        score += strategy.get('resource_efficiency', 0) * self.config.resource_weight
        
        # Time efficiency
        score += strategy.get('time_efficiency', 0) * self.config.time_weight
        
        # Complexity penalty
        score -= strategy.get('complexity', 0) * self.config.complexity_weight
        
        return score
    
    def _generate_execution_plan(self, 
                               ranked_strategies: List[Dict[str, any]]) -> Dict[str, any]:
        """Generate detailed execution plan"""
        return {
            'phases': self._generate_execution_phases(ranked_strategies),
            'dependencies': self._identify_strategy_dependencies(ranked_strategies),
            'resource_allocation': self._plan_resource_allocation(ranked_strategies),
            'timeline': self._generate_timeline(ranked_strategies),
            'monitoring_plan': self._generate_monitoring_plan(ranked_strategies)
        }
    
    def _estimate_required_resources(self, 
                                   strategies: List[Dict[str, any]]) -> Dict[str, any]:
        """Estimate required resources for recovery"""
        return {
            'computational': self._estimate_computational_resources(strategies),
            'memory': self._estimate_memory_requirements(strategies),
            'storage': self._estimate_storage_requirements(strategies),
            'network': self._estimate_network_requirements(strategies),
            'time': self._estimate_time_requirements(strategies)
        }
    
    def _estimate_success_probability(self, 
                                    strategies: List[Dict[str, any]]) -> Dict[str, float]:
        """Estimate success probability for different approaches"""
        return {
            'overall': self._calculate_overall_probability(strategies),
            'per_strategy': self._calculate_per_strategy_probability(strategies),
            'cumulative': self._calculate_cumulative_probability(strategies)
        }
    
    def _generate_recommendations(self, 
                                strategies: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Strategy-specific recommendations
        recommendations.extend(self._generate_strategy_recommendations(strategies))
        
        # Resource recommendations
        recommendations.extend(self._generate_resource_recommendations(strategies))
        
        # Optimization recommendations
        recommendations.extend(self._generate_optimization_recommendations(strategies))
        
        return sorted(recommendations, key=lambda x: x.get('priority', 0), reverse=True)