from typing import Dict, List, Optional, Tuple, Generator
from queue import PriorityQueue
import threading
import time
from datetime import datetime
from ..utils.logging import setup_logger

class Task:
    """Represents a recovery task with priority and dependencies"""
    
    def __init__(self, 
                 task_id: str,
                 task_type: str,
                 priority: int,
                 dependencies: List[str],
                 config: Dict[str, any]):
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
        self.dependencies = dependencies
        self.config = config
        self.status = 'pending'
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first

class TaskScheduler:
    """Manages task scheduling and load balancing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.task_queue = PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_history = []
        self.lock = threading.Lock()
        self.task_metrics = {
            'type_duration': {},
            'type_count': {},
            'success_rate': {},
            'resource_usage': {}
        }
        
    def submit_task(self, task: Task) -> str:
        """Submit a new task for execution"""
        with self.lock:
            # Validate dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    raise ValueError(f"Dependency {dep_id} not completed")
            
            # Add to queue
            self.task_queue.put(task)
            self.logger.info(f"Task {task.task_id} submitted with priority {task.priority}")
            
            return task.task_id
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute"""
        with self.lock:
            if self.task_queue.empty():
                return None
            
            # Get highest priority task with satisfied dependencies
            candidates = []
            while not self.task_queue.empty():
                task = self.task_queue.get()
                if self._are_dependencies_met(task):
                    self.active_tasks[task.task_id] = task
                    task.started_at = datetime.now()
                    self.logger.info(f"Task {task.task_id} started")
                    return task
                candidates.append(task)
            
            # Return unmet tasks to queue
            for task in candidates:
                self.task_queue.put(task)
            
            return None
    
    def complete_task(self, 
                     task_id: str,
                     result: Optional[any] = None,
                     error: Optional[str] = None):
        """Mark a task as completed"""
        with self.lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"Task {task_id} not found in active tasks")
            
            task = self.active_tasks[task_id]
            task.completed_at = datetime.now()
            task.result = result
            task.error = error
            task.status = 'completed' if error is None else 'failed'
            
            # Update metrics
            self._update_task_metrics(task)
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            self.task_history.append({
                'task_id': task_id,
                'type': task.task_type,
                'priority': task.priority,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'duration': (task.completed_at - task.started_at).total_seconds(),
                'status': task.status,
                'error': error
            })
            
            self.logger.info(f"Task {task_id} completed with status {task.status}")
    
    def get_task_status(self, task_id: str) -> Dict[str, any]:
        """Get current status of a task"""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                status = 'running'
            elif task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                status = task.status
            else:
                # Check queue
                found = False
                temp_queue = PriorityQueue()
                while not self.task_queue.empty():
                    t = self.task_queue.get()
                    if t.task_id == task_id:
                        task = t
                        status = 'pending'
                        found = True
                    temp_queue.put(t)
                
                # Restore queue
                while not temp_queue.empty():
                    self.task_queue.put(temp_queue.get())
                    
                if not found:
                    raise ValueError(f"Task {task_id} not found")
            
            return {
                'task_id': task.task_id,
                'type': task.task_type,
                'priority': task.priority,
                'status': status,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'result': task.result,
                'error': task.error
            }
    
    def get_metrics(self) -> Dict[str, any]:
        """Get current task metrics"""
        with self.lock:
            return {
                'queue_length': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'type_metrics': self.task_metrics,
                'success_rate': self._calculate_success_rate(),
                'average_duration': self._calculate_average_duration(),
                'throughput': self._calculate_throughput()
            }
    
    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are met"""
        return all(dep_id in self.completed_tasks 
                  and self.completed_tasks[dep_id].status == 'completed'
                  for dep_id in task.dependencies)
    
    def _update_task_metrics(self, task: Task):
        """Update task metrics after completion"""
        task_type = task.task_type
        duration = (task.completed_at - task.started_at).total_seconds()
        
        # Update type duration
        if task_type not in self.task_metrics['type_duration']:
            self.task_metrics['type_duration'][task_type] = []
        self.task_metrics['type_duration'][task_type].append(duration)
        
        # Update type count
        self.task_metrics['type_count'][task_type] = \
            self.task_metrics['type_count'].get(task_type, 0) + 1
        
        # Update success rate
        success = task.status == 'completed'
        if task_type not in self.task_metrics['success_rate']:
            self.task_metrics['success_rate'][task_type] = {'success': 0, 'total': 0}
        self.task_metrics['success_rate'][task_type]['total'] += 1
        if success:
            self.task_metrics['success_rate'][task_type]['success'] += 1