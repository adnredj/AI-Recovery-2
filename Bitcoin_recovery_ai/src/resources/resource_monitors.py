import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import threading
import time
from ..utils.logging import setup_logger

class BaseMonitor:
    """Base class for resource monitors"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.monitoring = False
        self.history = []
        self.lock = threading.Lock()
        self.monitor_thread = None
        
    def start(self):
        """Start resource monitoring"""
        with self.lock:
            if not self.monitoring:
                self.monitoring = True
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self.monitor_thread.start()
                self.logger.info(f"Started {self.__class__.__name__}")
    
    def stop(self):
        """Stop resource monitoring"""
        with self.lock:
            if self.monitoring:
                self.monitoring = False
                if self.monitor_thread:
                    self.monitor_thread.join()
                self.logger.info(f"Stopped {self.__class__.__name__}")
    
    def get_status(self) -> Dict[str, any]:
        """Get current resource status"""
        with self.lock:
            return self._get_current_status()
    
    def get_history(self) -> List[Dict[str, any]]:
        """Get monitoring history"""
        with self.lock:
            return self.history.copy()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                status = self._get_current_status()
                with self.lock:
                    self.history.append(status)
                    
                    # Trim history if needed
                    if len(self.history) > self.config.max_history_length:
                        self.history = self.history[-self.config.max_history_length:]
                        
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                
    def _get_current_status(self) -> Dict[str, any]:
        """Get current resource status - to be implemented by subclasses"""
        raise NotImplementedError

class CPUMonitor(BaseMonitor):
    """CPU resource monitoring"""
    
    def _get_current_status(self) -> Dict[str, any]:
        """Get current CPU status"""
        cpu_times = psutil.cpu_times_percent()
        cpu_freq = psutil.cpu_freq()
        
        return {
            'timestamp': time.time(),
            'usage_percent': psutil.cpu_percent(percpu=True),
            'average_load': np.mean(psutil.getloadavg()),
            'frequency': {
                'current': cpu_freq.current if cpu_freq else None,
                'min': cpu_freq.min if cpu_freq else None,
                'max': cpu_freq.max if cpu_freq else None
            },
            'times': {
                'user': cpu_times.user,
                'system': cpu_times.system,
                'idle': cpu_times.idle,
                'iowait': cpu_times.iowait if hasattr(cpu_times, 'iowait') else None
            },
            'core_count': psutil.cpu_count(logical=False),
            'thread_count': psutil.cpu_count(logical=True)
        }

class MemoryMonitor(BaseMonitor):
    """Memory resource monitoring"""
    
    def _get_current_status(self) -> Dict[str, any]:
        """Get current memory status"""
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        return {
            'timestamp': time.time(),
            'virtual_memory': {
                'total': virtual_memory.total,
                'available': virtual_memory.available,
                'used': virtual_memory.used,
                'free': virtual_memory.free,
                'percent': virtual_memory.percent,
                'cached': virtual_memory.cached if hasattr(virtual_memory, 'cached') else None
            },
            'swap_memory': {
                'total': swap_memory.total,
                'used': swap_memory.used,
                'free': swap_memory.free,
                'percent': swap_memory.percent
            }
        }

class GPUMonitor(BaseMonitor):
    """GPU resource monitoring"""
    
    def __init__(self, config):
        super().__init__(config)
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
        
    def _get_current_status(self) -> Dict[str, any]:
        """Get current GPU status"""
        if not self.gpu_available:
            return {
                'timestamp': time.time(),
                'available': False
            }
            
        gpu_stats = []
        for i in range(self.gpu_count):
            gpu_stats.append({
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'memory': {
                    'allocated': torch.cuda.memory_allocated(i),
                    'cached': torch.cuda.memory_cached(i),
                    'max': torch.cuda.max_memory_allocated(i)
                },
                'utilization': self._get_gpu_utilization(i)
            })
            
        return {
            'timestamp': time.time(),
            'available': True,
            'device_count': self.gpu_count,
            'devices': gpu_stats
        }
        
    def _get_gpu_utilization(self, device_id: int) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            # Try to get GPU utilization using nvidia-smi
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return None

class DiskMonitor(BaseMonitor):
    """Disk resource monitoring"""
    
    def _get_current_status(self) -> Dict[str, any]:
        """Get current disk status"""
        disk_usage = {}
        disk_io = psutil.disk_io_counters(perdisk=True)
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                }
            except Exception as e:
                self.logger.warning(f"Failed to get disk usage for {partition.mountpoint}: {str(e)}")
        
        return {
            'timestamp': time.time(),
            'partitions': disk_usage,
            'io_counters': {
                disk: {
                    'read_bytes': io.read_bytes,
                    'write_bytes': io.write_bytes,
                    'read_count': io.read_count,
                    'write_count': io.write_count,
                    'read_time': io.read_time,
                    'write_time': io.write_time
                }
                for disk, io in disk_io.items()
            }
        }

class NetworkMonitor(BaseMonitor):
    """Network resource monitoring"""
    
    def _get_current_status(self) -> Dict[str, any]:
        """Get current network status"""
        net_io = psutil.net_io_counters()
        net_connections = psutil.net_connections()
        
        return {
            'timestamp': time.time(),
            'io_counters': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            },
            'connection_count': len(net_connections),
            'connection_stats': self._analyze_connections(net_connections)
        }
        
    def _analyze_connections(self, connections: List[psutil._common.sconn]) -> Dict[str, int]:
        """Analyze network connections"""
        stats = {
            'ESTABLISHED': 0,
            'LISTEN': 0,
            'TIME_WAIT': 0,
            'CLOSE_WAIT': 0,
            'CLOSED': 0,
            'OTHER': 0
        }
        
        for conn in connections:
            status = conn.status if hasattr(conn, 'status') else 'OTHER'
            if status in stats:
                stats[status] += 1
            else:
                stats['OTHER'] += 1
                
        return stats