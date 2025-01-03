import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

class ResourcePlotter:
    """Generate visualizations for resource monitoring data"""
    
    def __init__(self, config):
        self.config = config
        self.style = self._setup_plot_style()
        
    def plot_cpu_usage(self, 
                      cpu_history: List[Dict[str, any]], 
                      save_path: Path) -> Path:
        """Plot CPU usage over time"""
        plt.figure(figsize=(12, 6))
        
        # Extract data
        timestamps = [entry['timestamp'] for entry in cpu_history]
        usage = [entry['usage_percent'] for entry in cpu_history]
        
        # Convert to datetime
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Plot overall CPU usage
        if isinstance(usage[0], list):
            # Multi-core data
            usage_array = np.array(usage)
            plt.plot(times, usage_array.mean(axis=1), label='Average', linewidth=2)
            plt.fill_between(times, 
                           usage_array.min(axis=1), 
                           usage_array.max(axis=1), 
                           alpha=0.3)
        else:
            # Single value data
            plt.plot(times, usage, label='Usage', linewidth=2)
            
        plt.title('CPU Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Usage (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_memory_usage(self,
                         memory_history: List[Dict[str, any]],
                         save_path: Path) -> Path:
        """Plot memory usage over time"""
        plt.figure(figsize=(12, 6))
        
        # Extract data
        timestamps = [entry['timestamp'] for entry in memory_history]
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        virtual_mem = [entry['virtual_memory'] for entry in memory_history]
        swap_mem = [entry['swap_memory'] for entry in memory_history]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Virtual memory plot
        ax1.plot(times, [vm['percent'] for vm in virtual_mem], 
                label='Usage', color='blue', linewidth=2)
        ax1.set_title('Virtual Memory Usage')
        ax1.set_ylabel('Usage (%)')
        ax1.grid(True, alpha=0.3)
        
        # Swap memory plot
        ax2.plot(times, [sm['percent'] for sm in swap_mem],
                label='Swap', color='red', linewidth=2)
        ax2.set_title('Swap Memory Usage')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Usage (%)')
        ax2.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_gpu_usage(self,
                      gpu_history: List[Dict[str, any]],
                      save_path: Path) -> Path:
        """Plot GPU usage over time"""
        if not gpu_history[0].get('available', False):
            return None
            
        # Create figure with subplots for each GPU
        device_count = gpu_history[0]['device_count']
        fig, axes = plt.subplots(device_count, 2, figsize=(15, 5*device_count))
        
        timestamps = [entry['timestamp'] for entry in gpu_history]
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        for device_idx in range(device_count):
            # Memory usage plot
            memory_data = [
                entry['devices'][device_idx]['memory']
                for entry in gpu_history
            ]
            
            axes[device_idx, 0].plot(times,
                                   [m['allocated']/m['max']*100 for m in memory_data],
                                   label='Memory Usage',
                                   color='blue',
                                   linewidth=2)
            axes[device_idx, 0].set_title(f'GPU {device_idx} Memory Usage')
            axes[device_idx, 0].set_ylabel('Usage (%)')
            axes[device_idx, 0].grid(True, alpha=0.3)
            
            # Utilization plot
            util_data = [
                entry['devices'][device_idx]['utilization']
                for entry in gpu_history
            ]
            
            axes[device_idx, 1].plot(times,
                                   util_data,
                                   label='Utilization',
                                   color='green',
                                   linewidth=2)
            axes[device_idx, 1].set_title(f'GPU {device_idx} Utilization')
            axes[device_idx, 1].set_ylabel('Utilization (%)')
            axes[device_idx, 1].grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_disk_usage(self,
                       disk_history: List[Dict[str, any]],
                       save_path: Path) -> Path:
        """Plot disk usage and I/O over time"""
        # Create subplots for usage and I/O
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        timestamps = [entry['timestamp'] for entry in disk_history]
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Plot disk usage
        partitions = list(disk_history[0]['partitions'].keys())
        for partition in partitions:
            usage = [entry['partitions'][partition]['percent'] 
                    for entry in disk_history]
            ax1.plot(times, usage, label=partition, linewidth=2)
            
        ax1.set_title('Disk Usage by Partition')
        ax1.set_ylabel('Usage (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot I/O rates
        io_data = self._calculate_io_rates(disk_history)
        ax2.plot(times[1:], io_data['read_rate'],
                label='Read Rate', color='blue', linewidth=2)
        ax2.plot(times[1:], io_data['write_rate'],
                label='Write Rate', color='red', linewidth=2)
        
        ax2.set_title('Disk I/O Rates')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rate (MB/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def _setup_plot_style(self):
        """Setup matplotlib plot style"""
        plt.style.use('seaborn')
        return {
            'figure.figsize': (12, 6),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'grid.alpha': 0.3
        }
    
    def _calculate_io_rates(self, disk_history: List[Dict[str, any]]) -> Dict[str, List[float]]:
        """Calculate I/O rates from disk history"""
        timestamps = np.array([entry['timestamp'] for entry in disk_history])
        time_diff = np.diff(timestamps)
        
        total_read = np.array([
            sum(io['read_bytes'] for io in entry['io_counters'].values())
            for entry in disk_history
        ])
        total_write = np.array([
            sum(io['write_bytes'] for io in entry['io_counters'].values())
            for entry in disk_history
        ])
        
        read_rate = np.diff(total_read) / time_diff / (1024 * 1024)  # MB/s
        write_rate = np.diff(total_write) / time_diff / (1024 * 1024)  # MB/s
        
        return {
            'read_rate': read_rate,
            'write_rate': write_rate
        }