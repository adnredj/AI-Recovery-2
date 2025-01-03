import time
from typing import Dict, List, Optional, Tuple, Generator
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from ..utils.logging import setup_logger
from ..utils.visualization import plot_utils
from ..utils.metrics import calculate_metrics

class RecoveryMonitor:
    """Monitor and analyze recovery progress"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.start_time = None
        self.metrics_history = []
        self.checkpoints = []
        
    def start_monitoring(self, strategy_info: Dict[str, any]):
        """Start monitoring recovery process"""
        self.start_time = datetime.now()
        self.strategy_info = strategy_info
        self.metrics_history = []
        self.checkpoints = []
        
        self.logger.info(f"Started monitoring recovery process at {self.start_time}")
        
    def update_progress(self, progress_data: Dict[str, any]) -> Dict[str, any]:
        """Update and analyze progress"""
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        
        # Calculate metrics
        metrics = self._calculate_current_metrics(progress_data, elapsed_time)
        self.metrics_history.append(metrics)
        
        # Analyze progress
        analysis = self._analyze_progress(metrics)
        
        # Check for significant events
        events = self._check_significant_events(metrics, analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        
        # Create checkpoint if needed
        if self._should_create_checkpoint(metrics):
            self._create_checkpoint(metrics, analysis)
        
        return {
            'timestamp': current_time,
            'elapsed_time': elapsed_time,
            'metrics': metrics,
            'analysis': analysis,
            'events': events,
            'recommendations': recommendations
        }
    
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive progress report"""
        if not self.metrics_history:
            return {'error': 'No monitoring data available'}
            
        return {
            'summary': self._generate_summary(),
            'detailed_metrics': self._generate_detailed_metrics(),
            'performance_analysis': self._analyze_performance(),
            'resource_utilization': self._analyze_resource_utilization(),
            'bottlenecks': self._identify_bottlenecks(),
            'optimization_opportunities': self._identify_optimizations(),
            'visualizations': self._generate_visualizations()
        }
    
    def _calculate_current_metrics(self, 
                                 progress_data: Dict[str, any],
                                 elapsed_time: timedelta) -> Dict[str, any]:
        """Calculate current performance metrics"""
        metrics = {
            'progress_percentage': progress_data.get('progress', 0) * 100,
            'processing_rate': self._calculate_processing_rate(progress_data),
            'success_rate': self._calculate_success_rate(progress_data),
            'resource_utilization': self._calculate_resource_utilization(),
            'efficiency_metrics': self._calculate_efficiency_metrics(progress_data),
            'time_metrics': {
                'elapsed_seconds': elapsed_time.total_seconds(),
                'estimated_remaining': self._estimate_remaining_time(progress_data)
            }
        }
        
        return metrics
    
    def _analyze_progress(self, metrics: Dict[str, any]) -> Dict[str, any]:
        """Analyze current progress and trends"""
        return {
            'performance_trends': self._analyze_performance_trends(),
            'efficiency_analysis': self._analyze_efficiency(),
            'resource_analysis': self._analyze_resource_usage(),
            'bottleneck_analysis': self._analyze_bottlenecks(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
    
    def _check_significant_events(self, 
                                metrics: Dict[str, any],
                                analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Check for significant events or milestones"""
        events = []
        
        # Check progress milestones
        if self._check_progress_milestone(metrics):
            events.append({
                'type': 'milestone',
                'description': f"Reached {metrics['progress_percentage']:.1f}% completion"
            })
        
        # Check performance changes
        perf_change = self._check_performance_change(metrics)
        if perf_change:
            events.append({
                'type': 'performance_change',
                'description': perf_change
            })
        
        # Check resource utilization
        resource_event = self._check_resource_event(metrics)
        if resource_event:
            events.append({
                'type': 'resource_event',
                'description': resource_event
            })
        
        # Check for potential issues
        issues = self._check_potential_issues(metrics, analysis)
        events.extend(issues)
        
        return events
    
    def _generate_recommendations(self, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance recommendations
        if analysis['performance_trends'].get('declining'):
            recommendations.extend(self._generate_performance_recommendations())
        
        # Resource recommendations
        if analysis['resource_analysis'].get('inefficient'):
            recommendations.extend(self._generate_resource_recommendations())
        
        # Optimization recommendations
        if analysis['optimization_opportunities']:
            recommendations.extend(self._generate_optimization_recommendations())
        
        return sorted(recommendations, key=lambda x: x.get('priority', 0), reverse=True)
    
    def _should_create_checkpoint(self, metrics: Dict[str, any]) -> bool:
        """Determine if checkpoint should be created"""
        if not self.checkpoints:
            return True
            
        last_checkpoint = self.checkpoints[-1]
        time_since_last = metrics['time_metrics']['elapsed_seconds'] - last_checkpoint['timestamp']
        
        return (time_since_last >= self.config.checkpoint_interval or
                self._significant_progress_since_last(metrics, last_checkpoint))
    
    def _create_checkpoint(self, 
                         metrics: Dict[str, any],
                         analysis: Dict[str, any]):
        """Create progress checkpoint"""
        checkpoint = {
            'timestamp': metrics['time_metrics']['elapsed_seconds'],
            'metrics': metrics.copy(),
            'analysis': analysis.copy(),
            'recovery_state': self._capture_recovery_state()
        }
        
        self.checkpoints.append(checkpoint)
        self.logger.info(f"Created checkpoint at {checkpoint['timestamp']} seconds")
    
    def _generate_visualizations(self) -> Dict[str, Path]:
        """Generate progress visualization plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_dir = Path(self.config.output_dir) / 'progress_plots' / timestamp
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # Progress over time plot
        plots['progress'] = plot_utils.plot_progress_over_time(
            self.metrics_history,
            plot_dir / 'progress.png'
        )
        
        # Performance metrics plot
        plots['performance'] = plot_utils.plot_performance_metrics(
            self.metrics_history,
            plot_dir / 'performance.png'
        )
        
        # Resource utilization plot
        plots['resources'] = plot_utils.plot_resource_utilization(
            self.metrics_history,
            plot_dir / 'resources.png'
        )
        
        return plots
    
    def _calculate_processing_rate(self, progress_data: Dict[str, any]) -> float:
        """Calculate current processing rate
        
        Args:
            progress_data: Dictionary containing progress information
            
        Returns:
            Current processing rate (items/second)
        """
        try:
            # Get current and previous metrics
            current_processed = progress_data.get('processed_items', 0)
            current_time = datetime.now()
            
            if not self.metrics_history:
                return 0.0
            
            last_metrics = self.metrics_history[-1]
            last_processed = last_metrics.get('processed_items', 0)
            last_time = last_metrics.get('timestamp')
            
            if not last_time:
                return 0.0
            
            # Calculate time difference
            time_diff = (current_time - last_time).total_seconds()
            if time_diff <= 0:
                return 0.0
            
            # Calculate processing rate
            items_diff = current_processed - last_processed
            rate = items_diff / time_diff
            
            # Apply smoothing using exponential moving average
            alpha = self.config.get('rate_smoothing_factor', 0.3)
            if 'smoothed_rate' in last_metrics:
                rate = alpha * rate + (1 - alpha) * last_metrics['smoothed_rate']
            
            return rate
            
        except Exception as e:
            self.logger.error(f"Error calculating processing rate: {str(e)}")
            return 0.0
    
    def _calculate_success_rate(self, progress_data: Dict[str, any]) -> float:
        """Calculate current success rate
        
        Args:
            progress_data: Dictionary containing progress information
            
        Returns:
            Current success rate (percentage)
        """
        try:
            total_attempts = progress_data.get('total_attempts', 0)
            successful_attempts = progress_data.get('successful_attempts', 0)
            
            if total_attempts == 0:
                return 0.0
            
            # Calculate basic success rate
            rate = (successful_attempts / total_attempts) * 100
            
            # Apply windowed averaging if configured
            window_size = self.config.get('success_rate_window', 1000)
            if len(self.metrics_history) >= 2:
                window_start = max(0, len(self.metrics_history) - window_size)
                window_metrics = self.metrics_history[window_start:]
                
                window_total = sum(m.get('total_attempts', 0) for m in window_metrics)
                window_success = sum(m.get('successful_attempts', 0) for m in window_metrics)
                
                if window_total > 0:
                    rate = (window_success / window_total) * 100
            
            return rate
            
        except Exception as e:
            self.logger.error(f"Error calculating success rate: {str(e)}")
            return 0.0
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization
        
        Returns:
            Dictionary containing resource utilization metrics
        """
        try:
            import psutil
            import GPUtil
            
            metrics = {
                'cpu': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent,
                'network': self._calculate_network_usage()
            }
            
            # Add GPU metrics if available
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics.update({
                        'gpu_load': np.mean([gpu.load * 100 for gpu in gpus]),
                        'gpu_memory': np.mean([gpu.memoryUtil * 100 for gpu in gpus])
                    })
            except Exception:
                pass
            
            # Calculate composite scores
            metrics['overall'] = np.mean([
                metrics[k] for k in ['cpu', 'memory', 'disk']
            ])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating resource utilization: {str(e)}")
            return {}
    
    def _calculate_efficiency_metrics(self, progress_data: Dict[str, any]) -> Dict[str, float]:
        """Calculate efficiency metrics
        
        Args:
            progress_data: Dictionary containing progress information
            
        Returns:
            Dictionary containing efficiency metrics
        """
        try:
            metrics = {}
            
            # Calculate processing efficiency
            target_rate = self.config.get('target_processing_rate', 1000)
            current_rate = self._calculate_processing_rate(progress_data)
            metrics['processing_efficiency'] = min(current_rate / target_rate * 100, 100)
            
            # Calculate resource efficiency
            resource_util = self._calculate_resource_utilization()
            metrics['resource_efficiency'] = 100 - resource_util.get('overall', 0)
            
            # Calculate success efficiency
            target_success = self.config.get('target_success_rate', 1.0)
            current_success = self._calculate_success_rate(progress_data)
            metrics['success_efficiency'] = min(current_success / target_success * 100, 100)
            
            # Calculate overall efficiency
            weights = self.config.get('efficiency_weights', {
                'processing': 0.4,
                'resource': 0.3,
                'success': 0.3
            })
            
            metrics['overall'] = sum(
                metrics[k] * weights.get(k.split('_')[0], 0)
                for k in metrics
                if k != 'overall'
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency metrics: {str(e)}")
            return {}
    
    def _estimate_remaining_time(self, progress_data: Dict[str, any]) -> float:
        """Estimate remaining time based on current progress
        
        Args:
            progress_data: Dictionary containing progress information
            
        Returns:
            Estimated remaining time in seconds
        """
        try:
            current_progress = progress_data.get('progress', 0)
            if current_progress <= 0:
                return float('inf')
            
            # Calculate based on recent processing rate
            current_rate = self._calculate_processing_rate(progress_data)
            if current_rate <= 0:
                return float('inf')
            
            remaining_items = progress_data.get('total_items', 0) - progress_data.get('processed_items', 0)
            basic_estimate = remaining_items / current_rate
            
            # Apply adjustments based on efficiency trends
            efficiency_metrics = self._calculate_efficiency_metrics(progress_data)
            efficiency_factor = efficiency_metrics.get('overall', 100) / 100
            
            # Apply additional factors from configuration
            adjustment_factor = self.config.get('time_estimate_adjustment', 1.2)
            
            adjusted_estimate = basic_estimate / efficiency_factor * adjustment_factor
            
            return max(0.0, adjusted_estimate)
            
        except Exception as e:
            self.logger.error(f"Error estimating remaining time: {str(e)}")
            return float('inf')
    
    def _calculate_network_usage(self) -> float:
        """Calculate network usage percentage"""
        try:
            import psutil
            
            # Get network IO counters
            net_io = psutil.net_io_counters()
            time.sleep(1)
            net_io_after = psutil.net_io_counters()
            
            # Calculate bytes per second
            bytes_sent = net_io_after.bytes_sent - net_io.bytes_sent
            bytes_recv = net_io_after.bytes_recv - net_io.bytes_recv
            
            # Convert to percentage based on typical network capacity
            typical_capacity = self.config.get('typical_network_capacity', 1000000)  # 1 MB/s
            usage_percentage = ((bytes_sent + bytes_recv) / typical_capacity) * 100
            
            return min(usage_percentage, 100)
            
        except Exception:
            return 0.0