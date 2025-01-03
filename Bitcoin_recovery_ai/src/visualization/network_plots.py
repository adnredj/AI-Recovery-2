import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

class NetworkPlotter:
    """Generate visualizations for network monitoring data"""
    
    def __init__(self, config):
        self.config = config
        self.style = self._setup_plot_style()
        
    def plot_network_traffic(self,
                           network_history: List[Dict[str, any]],
                           save_path: Path) -> Path:
        """Plot network traffic over time"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots for different metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Extract timestamps
        timestamps = [entry['timestamp'] for entry in network_history]
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Calculate rates
        traffic_rates = self._calculate_traffic_rates(network_history)
        
        # Plot bandwidth usage
        ax1.plot(times[1:], traffic_rates['send_rate'],
                label='Upload', color='blue', linewidth=2)
        ax1.plot(times[1:], traffic_rates['recv_rate'],
                label='Download', color='green', linewidth=2)
        ax1.set_title('Network Bandwidth Usage')
        ax1.set_ylabel('Rate (MB/s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot packet rates
        ax2.plot(times[1:], traffic_rates['packet_send_rate'],
                label='Packets Sent', color='blue', linewidth=2)
        ax2.plot(times[1:], traffic_rates['packet_recv_rate'],
                label='Packets Received', color='green', linewidth=2)
        ax2.set_title('Network Packet Rates')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Packets/s')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_connection_stats(self,
                            network_history: List[Dict[str, any]],
                            save_path: Path) -> Path:
        """Plot network connection statistics"""
        plt.figure(figsize=(12, 8))
        
        # Extract data
        timestamps = [entry['timestamp'] for entry in network_history]
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Get connection stats
        stats_keys = ['ESTABLISHED', 'LISTEN', 'TIME_WAIT', 'CLOSE_WAIT', 'CLOSED']
        stats_data = {
            key: [entry['connection_stats'].get(key, 0) 
                 for entry in network_history]
            for key in stats_keys
        }
        
        # Create stacked area plot
        plt.stackplot(times,
                     [stats_data[key] for key in stats_keys],
                     labels=stats_keys,
                     alpha=0.7)
        
        plt.title('Network Connection States')
        plt.xlabel('Time')
        plt.ylabel('Number of Connections')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_error_rates(self,
                        network_history: List[Dict[str, any]],
                        save_path: Path) -> Path:
        """Plot network error and drop rates"""
        plt.figure(figsize=(12, 6))
        
        # Extract timestamps
        timestamps = [entry['timestamp'] for entry in network_history]
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Calculate error rates
        error_rates = self._calculate_error_rates(network_history)
        
        # Plot error rates
        plt.plot(times[1:], error_rates['error_in_rate'],
                label='Input Errors', color='red', linewidth=2)
        plt.plot(times[1:], error_rates['error_out_rate'],
                label='Output Errors', color='orange', linewidth=2)
        plt.plot(times[1:], error_rates['drop_in_rate'],
                label='Input Drops', color='purple', linewidth=2)
        plt.plot(times[1:], error_rates['drop_out_rate'],
                label='Output Drops', color='brown', linewidth=2)
        
        plt.title('Network Error and Drop Rates')
        plt.xlabel('Time')
        plt.ylabel('Errors/s')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def generate_network_report(self,
                              network_history: List[Dict[str, any]],
                              save_dir: Path) -> Dict[str, Path]:
        """Generate comprehensive network analysis report"""
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plots = {}
        
        # Generate all plots
        plots['traffic'] = self.plot_network_traffic(
            network_history,
            save_dir / f'network_traffic_{timestamp}.png'
        )
        
        plots['connections'] = self.plot_connection_stats(
            network_history,
            save_dir / f'connection_stats_{timestamp}.png'
        )
        
        plots['errors'] = self.plot_error_rates(
            network_history,
            save_dir / f'error_rates_{timestamp}.png'
        )
        
        # Generate summary statistics
        stats = self._calculate_summary_statistics(network_history)
        stats_path = save_dir / f'network_stats_{timestamp}.json'
        self._save_json_stats(stats, stats_path)
        plots['stats'] = stats_path
        
        return plots
    
    def _calculate_traffic_rates(self,
                               network_history: List[Dict[str, any]]) -> Dict[str, np.ndarray]:
        """Calculate network traffic rates"""
        timestamps = np.array([entry['timestamp'] for entry in network_history])
        time_diff = np.diff(timestamps)
        
        # Extract counters
        bytes_sent = np.array([entry['io_counters']['bytes_sent'] 
                             for entry in network_history])
        bytes_recv = np.array([entry['io_counters']['bytes_recv'] 
                             for entry in network_history])
        packets_sent = np.array([entry['io_counters']['packets_sent'] 
                               for entry in network_history])
        packets_recv = np.array([entry['io_counters']['packets_recv'] 
                               for entry in network_history])
        
        # Calculate rates
        return {
            'send_rate': np.diff(bytes_sent) / time_diff / (1024 * 1024),  # MB/s
            'recv_rate': np.diff(bytes_recv) / time_diff / (1024 * 1024),  # MB/s
            'packet_send_rate': np.diff(packets_sent) / time_diff,  # packets/s
            'packet_recv_rate': np.diff(packets_recv) / time_diff   # packets/s
        }
    
    def _calculate_error_rates(self,
                             network_history: List[Dict[str, any]]) -> Dict[str, np.ndarray]:
        """Calculate network error rates"""
        timestamps = np.array([entry['timestamp'] for entry in network_history])
        time_diff = np.diff(timestamps)
        
        # Extract error counters
        errin = np.array([entry['io_counters']['errin'] for entry in network_history])
        errout = np.array([entry['io_counters']['errout'] for entry in network_history])
        dropin = np.array([entry['io_counters']['dropin'] for entry in network_history])
        dropout = np.array([entry['io_counters']['dropout'] for entry in network_history])
        
        # Calculate rates
        return {
            'error_in_rate': np.diff(errin) / time_diff,
            'error_out_rate': np.diff(errout) / time_diff,
            'drop_in_rate': np.diff(dropin) / time_diff,
            'drop_out_rate': np.diff(dropout) / time_diff
        }
    
    def _calculate_summary_statistics(self,
                                   network_history: List[Dict[str, any]]) -> Dict[str, any]:
        """Calculate summary statistics for network usage"""
        traffic_rates = self._calculate_traffic_rates(network_history)
        error_rates = self._calculate_error_rates(network_history)
        
        return {
            'traffic': {
                'avg_upload_rate': float(np.mean(traffic_rates['send_rate'])),
                'avg_download_rate': float(np.mean(traffic_rates['recv_rate'])),
                'peak_upload_rate': float(np.max(traffic_rates['send_rate'])),
                'peak_download_rate': float(np.max(traffic_rates['recv_rate']))
            },
            'errors': {
                'total_input_errors': int(np.sum(error_rates['error_in_rate'])),
                'total_output_errors': int(np.sum(error_rates['error_out_rate'])),
                'total_input_drops': int(np.sum(error_rates['drop_in_rate'])),
                'total_output_drops': int(np.sum(error_rates['drop_out_rate']))
            },
            'connections': {
                'avg_active_connections': float(np.mean([
                    entry['connection_count'] for entry in network_history
                ]))
            }
        }
    
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
    
    def _save_json_stats(self, stats: Dict[str, any], path: Path):
        """Save statistics to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(stats, f, indent=4)