import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from ..utils.logging import setup_logger
from ..utils.metrics import calculate_metrics

class PerformanceAnalyzer:
    """Analyze and optimize recovery performance"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.metrics_history = []
        self.optimization_history = []
        
    def analyze_performance(self, 
                          metrics: Dict[str, any],
                          resource_data: Dict[str, any]) -> Dict[str, any]:
        """Analyze current performance metrics"""
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'resources': resource_data
        })
        
        # Perform analysis
        analysis = {
            'throughput': self._analyze_throughput(metrics),
            'efficiency': self._analyze_efficiency(metrics, resource_data),
            'bottlenecks': self._identify_bottlenecks(metrics, resource_data),
            'resource_utilization': self._analyze_resource_utilization(resource_data),
            'optimization_opportunities': self._identify_optimization_opportunities(
                metrics, resource_data
            )
        }
        
        # Calculate performance scores
        analysis['scores'] = self._calculate_performance_scores(analysis)
        
        return analysis
    
    def generate_optimization_plan(self, 
                                 analysis: Dict[str, any]) -> Dict[str, any]:
        """Generate performance optimization plan"""
        
        plan = {
            'timestamp': datetime.now(),
            'recommendations': [],
            'priority_actions': [],
            'expected_improvements': {}
        }
        
        # Generate recommendations based on bottlenecks
        for bottleneck in analysis['bottlenecks']:
            recommendations = self._generate_bottleneck_recommendations(bottleneck)
            plan['recommendations'].extend(recommendations)
        
        # Generate optimization opportunities
        for opportunity in analysis['optimization_opportunities']:
            optimization = self._generate_optimization_action(opportunity)
            if optimization['priority'] >= self.config.optimization_priority_threshold:
                plan['priority_actions'].append(optimization)
            else:
                plan['recommendations'].append(optimization)
        
        # Calculate expected improvements
        plan['expected_improvements'] = self._estimate_improvements(
            analysis,
            plan['priority_actions']
        )
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'analysis': analysis,
            'plan': plan
        })
        
        return plan
    
    def _analyze_throughput(self, metrics: Dict[str, any]) -> Dict[str, any]:
        """Analyze processing throughput"""
        current_throughput = metrics['processing_rate']
        historical_throughput = [m['metrics']['processing_rate'] 
                               for m in self.metrics_history]
        
        return {
            'current_rate': current_throughput,
            'average_rate': np.mean(historical_throughput),
            'peak_rate': np.max(historical_throughput),
            'trend': self._calculate_trend(historical_throughput),
            'stability': self._calculate_stability(historical_throughput)
        }
    
    def _analyze_efficiency(self,
                          metrics: Dict[str, any],
                          resource_data: Dict[str, any]) -> Dict[str, any]:
        """Analyze processing efficiency"""
        
        # Calculate efficiency metrics
        cpu_efficiency = self._calculate_cpu_efficiency(metrics, resource_data)
        memory_efficiency = self._calculate_memory_efficiency(metrics, resource_data)
        gpu_efficiency = self._calculate_gpu_efficiency(metrics, resource_data)
        
        return {
            'cpu_efficiency': cpu_efficiency,
            'memory_efficiency': memory_efficiency,
            'gpu_efficiency': gpu_efficiency,
            'overall_efficiency': np.mean([
                cpu_efficiency['score'],
                memory_efficiency['score'],
                gpu_efficiency['score']
            ])
        }
    
    def _identify_bottlenecks(self,
                            metrics: Dict[str, any],
                            resource_data: Dict[str, any]) -> List[Dict[str, any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Check CPU bottlenecks
        if self._is_cpu_bottleneck(metrics, resource_data):
            bottlenecks.append({
                'type': 'cpu',
                'severity': self._calculate_bottleneck_severity(
                    resource_data['cpu']['usage_percent']
                ),
                'details': self._analyze_cpu_bottleneck(metrics, resource_data)
            })
        
        # Check memory bottlenecks
        if self._is_memory_bottleneck(metrics, resource_data):
            bottlenecks.append({
                'type': 'memory',
                'severity': self._calculate_bottleneck_severity(
                    resource_data['memory']['usage_percent']
                ),
                'details': self._analyze_memory_bottleneck(metrics, resource_data)
            })
        
        # Check GPU bottlenecks
        if resource_data.get('gpu', {}).get('available', False):
            if self._is_gpu_bottleneck(metrics, resource_data):
                bottlenecks.append({
                    'type': 'gpu',
                    'severity': self._calculate_bottleneck_severity(
                        resource_data['gpu']['utilization']
                    ),
                    'details': self._analyze_gpu_bottleneck(metrics, resource_data)
                })
        
        return sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)
    
    def _analyze_resource_utilization(self,
                                    resource_data: Dict[str, any]) -> Dict[str, any]:
        """Analyze resource utilization patterns"""
        return {
            'cpu': self._analyze_cpu_utilization(resource_data),
            'memory': self._analyze_memory_utilization(resource_data),
            'gpu': self._analyze_gpu_utilization(resource_data),
            'overall': self._calculate_overall_utilization(resource_data)
        }
    
    def _identify_optimization_opportunities(self,
                                          metrics: Dict[str, any],
                                          resource_data: Dict[str, any]) -> List[Dict[str, any]]:
        """Identify potential optimization opportunities"""
        opportunities = []
        
        # Check for CPU optimization opportunities
        cpu_opts = self._identify_cpu_optimizations(metrics, resource_data)
        opportunities.extend(cpu_opts)
        
        # Check for memory optimization opportunities
        memory_opts = self._identify_memory_optimizations(metrics, resource_data)
        opportunities.extend(memory_opts)
        
        # Check for GPU optimization opportunities
        if resource_data.get('gpu', {}).get('available', False):
            gpu_opts = self._identify_gpu_optimizations(metrics, resource_data)
            opportunities.extend(gpu_opts)
        
        # Sort by potential impact
        return sorted(opportunities, 
                     key=lambda x: x['potential_improvement'],
                     reverse=True)
    
    def _calculate_performance_scores(self, analysis: Dict[str, any]) -> Dict[str, float]:
        """Calculate performance scores"""
        return {
            'throughput_score': self._calculate_throughput_score(
                analysis['throughput']
            ),
            'efficiency_score': analysis['efficiency']['overall_efficiency'],
            'resource_score': self._calculate_resource_score(
                analysis['resource_utilization']
            ),
            'overall_score': self._calculate_overall_score(analysis)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, any]:
        """Calculate trend in time series data"""
        if len(values) < 2:
            return {'direction': 'stable', 'magnitude': 0.0}
            
        # Calculate trend using linear regression
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        # Determine trend direction and magnitude
        if abs(slope) < self.config.trend_threshold:
            direction = 'stable'
        else:
            direction = 'increasing' if slope > 0 else 'decreasing'
            
        return {
            'direction': direction,
            'magnitude': abs(slope),
            'slope': slope
        }
    
    def _calculate_stability(self, values: List[float]) -> Dict[str, float]:
        """Calculate stability metrics for time series data"""
        if len(values) < 2:
            return {
                'std_dev': 0.0,
                'coefficient_variation': 0.0,
                'stability_score': 1.0
            }
            
        values = np.array(values)
        mean = np.mean(values)
        std_dev = np.std(values)
        
        return {
            'std_dev': std_dev,
            'coefficient_variation': std_dev / mean if mean != 0 else 0.0,
            'stability_score': 1.0 - min(std_dev / mean if mean != 0 else 0.0, 1.0)
        }