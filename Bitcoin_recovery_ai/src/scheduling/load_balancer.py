from typing import Dict, List, Optional, Tuple
import threading
import time
from datetime import datetime
import numpy as np
from ..utils.logging import setup_logger
from ..utils.metrics import calculate_metrics

class LoadBalancer:
    """Manages resource distribution and load balancing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.lock = threading.Lock()
        self.worker_stats = {}
        self.resource_allocation = {}
        self.load_history = []
        self.balancing_metrics = {
            'distribution_scores': [],
            'rebalance_counts': {},
            'efficiency_scores': []
        }
        
    def register_worker(self, 
                       worker_id: str,
                       capabilities: Dict[str, any]) -> Dict[str, any]:
        """Register a new worker with its capabilities"""
        with self.lock:
            self.worker_stats[worker_id] = {
                'capabilities': capabilities,
                'current_load': 0.0,
                'task_history': [],
                'performance_metrics': {},
                'status': 'active',
                'last_heartbeat': datetime.now()
            }
            
            # Calculate initial resource allocation
            allocation = self._calculate_initial_allocation(
                worker_id,
                capabilities
            )
            self.resource_allocation[worker_id] = allocation
            
            self.logger.info(f"Worker {worker_id} registered with capabilities: {capabilities}")
            return allocation
    
    def update_worker_status(self,
                           worker_id: str,
                           status_update: Dict[str, any]):
        """Update worker status and metrics"""
        with self.lock:
            if worker_id not in self.worker_stats:
                raise ValueError(f"Worker {worker_id} not registered")
                
            worker = self.worker_stats[worker_id]
            worker['last_heartbeat'] = datetime.now()
            worker['current_load'] = status_update.get('load', worker['current_load'])
            
            # Update performance metrics
            if 'metrics' in status_update:
                self._update_performance_metrics(worker_id, status_update['metrics'])
            
            # Check for rebalancing need
            if self._should_rebalance(worker_id):
                self._rebalance_resources()
    
    def assign_task(self,
                   task: Dict[str, any]) -> Optional[str]:
        """Assign task to most suitable worker"""
        with self.lock:
            # Get active workers
            active_workers = {
                worker_id: stats for worker_id, stats in self.worker_stats.items()
                if stats['status'] == 'active'
            }
            
            if not active_workers:
                return None
            
            # Calculate worker scores for this task
            worker_scores = self._calculate_worker_scores(task, active_workers)
            
            # Select best worker
            best_worker = max(worker_scores.items(), key=lambda x: x[1])[0]
            
            # Update worker stats
            self.worker_stats[best_worker]['current_load'] += task.get('load_estimate', 1.0)
            self.worker_stats[best_worker]['task_history'].append({
                'task_id': task['task_id'],
                'assigned_at': datetime.now()
            })
            
            self.logger.info(f"Task {task['task_id']} assigned to worker {best_worker}")
            return best_worker
    
    def get_system_load(self) -> Dict[str, any]:
        """Get current system load distribution"""
        with self.lock:
            current_load = {
                worker_id: stats['current_load']
                for worker_id, stats in self.worker_stats.items()
                if stats['status'] == 'active'
            }
            
            return {
                'worker_loads': current_load,
                'total_load': sum(current_load.values()),
                'load_distribution': self._calculate_load_distribution(),
                'resource_utilization': self._calculate_resource_utilization()
            }
    
    def _calculate_worker_scores(self,
                               task: Dict[str, any],
                               workers: Dict[str, Dict[str, any]]) -> Dict[str, float]:
        """Calculate suitability scores for workers"""
        scores = {}
        
        for worker_id, stats in workers.items():
            # Base score on current load (inverse relationship)
            load_score = 1.0 - stats['current_load']
            
            # Capability match score
            capability_score = self._calculate_capability_match(
                task.get('requirements', {}),
                stats['capabilities']
            )
            
            # Performance score based on history
            performance_score = self._calculate_performance_score(
                worker_id,
                task['task_type']
            )
            
            # Combine scores with weights from config
            scores[worker_id] = (
                self.config.load_weight * load_score +
                self.config.capability_weight * capability_score +
                self.config.performance_weight * performance_score
            )
            
        return scores
    
    def _calculate_capability_match(self,
                                  requirements: Dict[str, any],
                                  capabilities: Dict[str, any]) -> float:
        """Calculate how well worker capabilities match task requirements"""
        if not requirements:
            return 1.0
            
        matches = []
        for req_key, req_value in requirements.items():
            if req_key not in capabilities:
                matches.append(0.0)
                continue
                
            cap_value = capabilities[req_key]
            if isinstance(req_value, (int, float)):
                # Numeric comparison
                matches.append(min(cap_value / req_value, 1.0))
            else:
                # Boolean or string comparison
                matches.append(1.0 if cap_value == req_value else 0.0)
                
        return np.mean(matches)
    
    def _calculate_performance_score(self,
                                  worker_id: str,
                                  task_type: str) -> float:
        """Calculate worker performance score for task type"""
        metrics = self.worker_stats[worker_id]['performance_metrics']
        
        if task_type not in metrics:
            return 0.5  # Neutral score for unknown task types
            
        type_metrics = metrics[task_type]
        
        # Combine various performance metrics
        success_rate = type_metrics['success_rate']
        speed_score = type_metrics['speed_score']
        efficiency_score = type_metrics['efficiency_score']
        
        return np.mean([success_rate, speed_score, efficiency_score])
    
    def _should_rebalance(self, worker_id: str) -> bool:
        """Determine if resource rebalancing is needed"""
        current_distribution = self._calculate_load_distribution()
        threshold = self.config.load_imbalance_threshold
        
        return (max(current_distribution.values()) - 
                min(current_distribution.values())) > threshold
    
    def _rebalance_resources(self):
        """Rebalance resources across workers"""
        current_loads = {
            worker_id: stats['current_load']
            for worker_id, stats in self.worker_stats.items()
            if stats['status'] == 'active'
        }
        
        if not current_loads:
            return
            
        avg_load = np.mean(list(current_loads.values()))
        
        # Calculate new allocations
        new_allocations = {}
        for worker_id, stats in self.worker_stats.items():
            if stats['status'] != 'active':
                continue
                
            current_load = current_loads[worker_id]
            load_diff = avg_load - current_load
            
            # Adjust resource allocation based on load difference
            new_allocation = self._adjust_allocation(
                self.resource_allocation[worker_id],
                load_diff
            )
            new_allocations[worker_id] = new_allocation
        
        # Update allocations
        self.resource_allocation.update(new_allocations)
        
        # Record rebalancing event
        self.load_history.append({
            'timestamp': datetime.now(),
            'loads': current_loads.copy(),
            'allocations': new_allocations.copy()
        })
        
        self.logger.info("Resources rebalanced across workers")