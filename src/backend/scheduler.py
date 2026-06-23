import concurrent.futures
from collections import defaultdict, deque

class Task:
    def __init__(self, name: str, fn, args: tuple = (), dependencies: list[str] = None):
        self.name = name
        self.fn = fn
        self.args = args or ()
        self.dependencies = dependencies or []
        self.completed = False
        self.result = None

class TaskScheduler:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.tasks = {}  # task_name -> Task
        self.adjacency = defaultdict(list)  # task_name -> list of dependent task_names
        self.in_degree = {}  # task_name -> int

    def add_task(self, name: str, fn, args: tuple = (), dependencies: list[str] = None):
        task = Task(name, fn, args, dependencies)
        self.tasks[name] = task
        self.in_degree[name] = len(task.dependencies)
        for dep in task.dependencies:
            self.adjacency[dep].append(name)

    def run_scheduler(self) -> dict:
        """
        Schedules and runs tasks in parallel using a thread pool, honoring DAG dependencies.
        """
        results = {}
        # Find ready tasks (in_degree == 0)
        ready_queue = deque([name for name, degree in self.in_degree.items() if degree == 0])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_to_name = {}
            
            # Start ready tasks
            while ready_queue or futures_to_name:
                while ready_queue:
                    name = ready_queue.popleft()
                    task = self.tasks[name]
                    future = executor.submit(task.fn, *task.args)
                    futures_to_name[future] = name
                
                if futures_to_name:
                    # Wait for at least one future to complete
                    done, _ = concurrent.futures.wait(
                        futures_to_name.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        name = futures_to_name.pop(future)
                        task = self.tasks[name]
                        try:
                            res = future.result()
                            task.result = res
                            task.completed = True
                            results[name] = res
                        except Exception as e:
                            print(f"Error in task '{name}': {e}")
                            raise e
                            
                        # Decrement in_degrees of dependent tasks
                        for dependent in self.adjacency[name]:
                            self.in_degree[dependent] -= 1
                            if self.in_degree[dependent] == 0:
                                ready_queue.append(dependent)
        return results
