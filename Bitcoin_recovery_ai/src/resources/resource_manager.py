import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from ..utils.logging import setup_logger

class ResourceManager:
    """Manage and optimize resource allocation for recovery tasks"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.resource_lock = threading.Lock()
        self.allocated_resources = {}
        self.resource_usage_history = []
        
        # Initialize resource monitoring
        self.cpu_monitor = CPUMonitor(config)
        self.memory_monitor = MemoryMonitor(config)
        self.gpu_monitor = GPUMonitor(config)
        self.disk_monitor = DiskMonitor(config)
        
    def allocate_resources(self, task_id: str, requirements: Dict[str, any]) -> Dict[str, any]:
        """Allocate resources for a recovery task"""
        with self.resource_lock:
            # Check resource availability
            available_resources = self._get_available_resources()
            
            # Calculate optimal allocation
            allocation = self._calculate_optimal_allocation(
                requirements,
                available_resources
            )
            
            if not self._can_allocate(allocation):
                raise ResourceError("Insufficient resources available")
                
            # Perform allocation
            self._perform_allocation(task_id, allocation)
            
            # Update resource tracking
            self.allocated_resources[task_id] = allocation
            
            return allocation
    
    def release_resources(self, task_id: str):
        """Release resources allocated to a task"""
        with self.resource_lock:
            if task_id in self.allocated_resources:
                allocation = self.allocated_resources[task_id]
                self._release_allocation(allocation)
                del self.allocated_resources[task_id]
                
                self.logger.info(f"Released resources for task {task_id}")
    
    def optimize_resources(self) -> Dict[str, any]:
        """Optimize current resource allocation"""
        with self.resource_lock:
            # Analyze current usage
            usage_analysis = self._analyze_resource_usage()
            
            # Generate optimization plan
            optimization_plan = self._generate_optimization_plan(usage_analysis)
            
            # Apply optimizations
            if optimization_plan['should_optimize']:
                self._apply_optimizations(optimization_plan['optimizations'])
                
            return optimization_plan
    
    def get_resource_status(self) -> Dict[str, any]:
        """Get current resource status"""
        return {
            'cpu': self.cpu_monitor.get_status(),
            'memory': self.memory_monitor.get_status(),
            'gpu': self.gpu_monitor.get_status(),
            'disk': self.disk_monitor.get_status(),
            'allocated': self.allocated_resources.copy(),
            'available': self._get_available_resources()
        }
    
    def _get_available_resources(self) -> Dict[str, any]:
        """Get currently available resources"""
        return {
            'cpu': {
                'cores': self._get_available_cpu_cores(),
                'usage': self._get_cpu_usage()
            },
            'memory': {
                'available': self._get_available_memory(),
                'total': psutil.virtual_memory().total
            },
            'gpu': self._get_gpu_resources(),
            'disk': {
                'space': self._get_available_disk_space(),
                'io_capacity': self._get_io_capacity()
            }
        }
    
    def _calculate_optimal_allocation(self,
                                   requirements: Dict[str, any],
                                   available: Dict[str, any]) -> Dict[str, any]:
        """Calculate optimal resource allocation"""
        allocation = {}
        
        # CPU allocation
        allocation['cpu'] = self._calculate_cpu_allocation(
            requirements.get('cpu', {}),
            available['cpu']
        )
        
        # Memory allocation
        allocation['memory'] = self._calculate_memory_allocation(
            requirements.get('memory', {}),
            available['memory']
        )
        
        # GPU allocation
        if requirements.get('gpu'):
            allocation['gpu'] = self._calculate_gpu_allocation(
                requirements['gpu'],
                available['gpu']
            )
            
        # Disk allocation
        allocation['disk'] = self._calculate_disk_allocation(
            requirements.get('disk', {}),
            available['disk']
        )
        
        return allocation
    
    def _analyze_resource_usage(self) -> Dict[str, any]:
        """Analyze current resource usage patterns"""
        return {
            'usage_patterns': self._analyze_usage_patterns(),
            'bottlenecks': self._identify_resource_bottlenecks(),
            'efficiency': self._calculate_resource_efficiency(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
    
    def _generate_optimization_plan(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """Generate resource optimization plan"""
        plan = {
            'should_optimize': False,
            'optimizations': []
        }
        
        # Check for optimization opportunities
        if analysis['optimization_opportunities']:
            plan['should_optimize'] = True
            plan['optimizations'] = self._create_optimization_actions(
                analysis['optimization_opportunities']
            )
            
        return plan
    
    def _apply_optimizations(self, optimizations: List[Dict[str, any]]):
        """Apply resource optimizations"""
        for opt in optimizations:
            try:
                if opt['type'] == 'reallocation':
                    self._apply_reallocation(opt)
                elif opt['type'] == 'scaling':
                    self._apply_scaling(opt)
                elif opt['type'] == 'consolidation':
                    self._apply_consolidation(opt)
                    
                self.logger.info(f"Applied optimization: {opt['description']}")
                
            except Exception as e:
                self.logger.error(f"Failed to apply optimization: {str(e)}")
    
    def _calculate_cpu_allocation(self,
                                requirements: Dict[str, any],
                                available: Dict[str, any]) -> Dict[str, any]:
        """Calculate CPU allocation based on requirements and availability
        
        Args:
            requirements: CPU requirements dictionary
            available: Available CPU resources dictionary
            
        Returns:
            Dictionary containing CPU allocation details
        """
        try:
            allocation = {
                'cores': [],
                'threads': 0,
                'priority': None,
                'affinity': None
            }
            
            # Get requirements
            required_cores = requirements.get('cores', 1)
            required_threads = requirements.get('threads', required_cores)
            min_frequency = requirements.get('min_frequency_mhz', 0)
            
            # Get available cores
            available_cores = available.get('cores', [])
            if not available_cores:
                available_cores = list(range(psutil.cpu_count(logical=False)))
            
            # Filter cores based on frequency requirement
            if min_frequency > 0:
                cpu_freq = psutil.cpu_freq(percpu=True)
                available_cores = [
                    core for core in available_cores
                    if cpu_freq[core].current >= min_frequency
                ]
            
            if len(available_cores) < required_cores:
                raise ResourceError(
                    f"Insufficient CPU cores. Required: {required_cores}, Available: {len(available_cores)}"
                )
            
            # Select optimal cores based on current load and frequency
            selected_cores = self._select_optimal_cores(
                available_cores,
                required_cores,
                self.cpu_monitor.get_core_metrics()
            )
            
            # Calculate thread allocation
            threads_per_core = psutil.cpu_count() // psutil.cpu_count(logical=False)
            max_threads = len(selected_cores) * threads_per_core
            
            if required_threads > max_threads:
                raise ResourceError(
                    f"Insufficient CPU threads. Required: {required_threads}, Available: {max_threads}"
                )
            
            # Set allocation details
            allocation['cores'] = selected_cores
            allocation['threads'] = required_threads
            allocation['priority'] = self._calculate_process_priority(requirements)
            allocation['affinity'] = self._calculate_core_affinity(selected_cores)
            
            return allocation
            
        except Exception as e:
            raise ResourceError(f"CPU allocation error: {str(e)}")
    
    def _calculate_memory_allocation(self,
                                   requirements: Dict[str, any],
                                   available: Dict[str, any]) -> Dict[str, any]:
        """Calculate memory allocation based on requirements and availability
        
        Args:
            requirements: Memory requirements dictionary
            available: Available memory resources dictionary
            
        Returns:
            Dictionary containing memory allocation details
        """
        try:
            allocation = {
                'bytes': 0,
                'type': 'unified',
                'segments': [],
                'swap': 0
            }
            
            # Get requirements
            required_bytes = requirements.get('bytes', 0)
            required_contiguous = requirements.get('contiguous', False)
            allow_swap = requirements.get('allow_swap', True)
            
            # Check available memory
            available_bytes = available.get('available', 0)
            total_bytes = available.get('total', 0)
            
            if required_bytes > available_bytes:
                if not allow_swap:
                    raise ResourceError(
                        f"Insufficient memory. Required: {required_bytes}, Available: {available_bytes}"
                    )
                
                # Calculate swap allocation
                swap = psutil.swap_memory()
                available_swap = swap.free
                
                if required_bytes > (available_bytes + available_swap):
                    raise ResourceError("Insufficient memory even with swap")
                
                allocation['swap'] = required_bytes - available_bytes
                
            # Handle contiguous memory requirement
            if required_contiguous:
                segments = self._find_contiguous_segments(required_bytes)
                if not segments:
                    raise ResourceError("Cannot allocate contiguous memory")
                allocation['segments'] = segments
                allocation['type'] = 'contiguous'
            else:
                allocation['segments'] = [{'start': 0, 'size': required_bytes}]
            
            allocation['bytes'] = required_bytes
            
            return allocation
            
        except Exception as e:
            raise ResourceError(f"Memory allocation error: {str(e)}")
    
    def _calculate_gpu_allocation(self,
                                requirements: Dict[str, any],
                                available: Dict[str, any]) -> Dict[str, any]:
        """Calculate GPU allocation based on requirements and availability
        
        Args:
            requirements: GPU requirements dictionary
            available: Available GPU resources dictionary
            
        Returns:
            Dictionary containing GPU allocation details
        """
        try:
            allocation = {
                'devices': [],
                'memory_per_device': 0,
                'compute_mode': None,
                'power_limit': None
            }
            
            # Check if GPU is required
            if not requirements:
                return allocation
            
            # Verify GPU availability
            if not torch.cuda.is_available():
                raise ResourceError("GPU required but not available")
            
            # Get requirements
            required_devices = requirements.get('devices', 1)
            required_memory = requirements.get('memory_per_device', 0)
            required_compute = requirements.get('compute_capability', 0.0)
            
            # Get available devices
            available_devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_free = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                
                if (memory_free >= required_memory and 
                    props.major + props.minor/10 >= required_compute):
                    available_devices.append({
                        'index': i,
                        'memory_free': memory_free,
                        'compute_capability': props.major + props.minor/10,
                        'performance_state': self.gpu_monitor.get_performance_state(i)
                    })
            
            if len(available_devices) < required_devices:
                raise ResourceError(
                    f"Insufficient GPU devices. Required: {required_devices}, "
                    f"Available: {len(available_devices)}"
                )
            
            # Select optimal devices
            selected_devices = self._select_optimal_gpus(
                available_devices,
                required_devices,
                self.gpu_monitor.get_device_metrics()
            )
            
            # Set allocation details
            allocation['devices'] = [dev['index'] for dev in selected_devices]
            allocation['memory_per_device'] = required_memory
            allocation['compute_mode'] = self._determine_compute_mode(requirements)
            allocation['power_limit'] = self._calculate_power_limit(requirements)
            
            return allocation
            
        except Exception as e:
            raise ResourceError(f"GPU allocation error: {str(e)}")
    
    def _calculate_disk_allocation(self,
                                 requirements: Dict[str, any],
                                 available: Dict[str, any]) -> Dict[str, any]:
        """Calculate disk allocation based on requirements and availability
        
        Args:
            requirements: Disk requirements dictionary
            available: Available disk resources dictionary
            
        Returns:
            Dictionary containing disk allocation details
        """
        try:
            allocation = {
                'paths': [],
                'bytes_per_path': 0,
                'io_priority': None,
                'buffer_size': 0
            }
            
            # Get requirements
            required_bytes = requirements.get('bytes', 0)
            required_speed = requirements.get('min_speed_mbps', 0)
            buffer_size = requirements.get('buffer_size', 8192)
            
            # Check available space
            available_space = available.get('space', {})
            io_capacity = available.get('io_capacity', {})
            
            # Find suitable paths
            suitable_paths = []
            for path, space in available_space.items():
                if space >= required_bytes:
                    if required_speed <= 0 or io_capacity.get(path, 0) >= required_speed:
                        suitable_paths.append(path)
            
            if not suitable_paths:
                raise ResourceError("No suitable disk locations found")
            
            # Select optimal paths based on performance metrics
            selected_paths = self._select_optimal_paths(
                suitable_paths,
                required_bytes,
                self.disk_monitor.get_path_metrics()
            )
            
            # Set allocation details
            allocation['paths'] = selected_paths
            allocation['bytes_per_path'] = required_bytes // len(selected_paths)
            allocation['io_priority'] = self._calculate_io_priority(requirements)
            allocation['buffer_size'] = buffer_size
            
            return allocation
            
        except Exception as e:
            raise ResourceError(f"Disk allocation error: {str(e)}")

class ResourceError(Exception):
    """Custom exception for resource-related errors"""
    pass