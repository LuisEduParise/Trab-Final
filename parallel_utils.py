# parallel_utils.py
import multiprocessing as mp
from mip_problem import Constraint

class SharedState:
    """
    Gerencia o estado global compartilhado entre todos os processos workers.
    """
    def __init__(self, num_workers: int, sense: str = "minimize"):
        self.num_workers = num_workers #
        self.sense = sense #
        
        manager = mp.Manager() #
        
        initial_primal = float('inf') if self.sense == 'minimize' else -float('inf') #
        self.best_cost = mp.Value('d', initial_primal) #
        self.best_solution = manager.dict() 
        self.has_solution = mp.Value('b', False) #
        self.idle_workers = mp.Value('i', 0) #
        self.nodes_processed = mp.Value('i', 0) #
        self.last_update_node_count = mp.Value('i', 0) #
        
        self.worker_best_bounds = manager.dict({i: initial_primal for i in range(num_workers)}) #
        
        self.cut_pool = manager.list() #
        self.cut_lock = mp.Lock() #
        self.lock = mp.Lock() #

    # --- CORREÇÃO ESTÁ AQUI ---
    # A assinatura do método agora aceita 'self', 'cost', e 'solution' (3 argumentos no total).
    def update_best_solution(self, cost: float, solution: dict) -> bool:
        with self.lock: #
            is_better = False #
            if self.sense == 'minimize': #
                if cost < self.best_cost.value: is_better = True #
            else: #
                if cost > self.best_cost.value: is_better = True #
            
            if is_better: #
                self.best_cost.value = cost #
                
                # Atualiza o dicionário da solução de forma segura
                self.best_solution.clear()
                self.best_solution.update(solution)
                
                if not self.has_solution.value: self.has_solution.value = True #
                self.last_update_node_count.value = self.nodes_processed.value #
                return True #
        return False #

    def get_best_solution(self) -> dict:
        """Retorna uma cópia do dicionário da melhor solução."""
        if self.has_solution.value:
            return dict(self.best_solution)
        return {}

    def update_worker_best_bound(self, worker_id: int, bound: float):
        """Worker reporta seu melhor bound local (o topo de seu heap)."""
        self.worker_best_bounds[worker_id] = bound #
        
    def get_last_update_node(self) -> int:
        return self.last_update_node_count.value #

    def add_cuts(self, new_cuts: list):
        with self.cut_lock: #
            existing_cuts = set(self.cut_pool) #
            for cut in new_cuts: #
                if cut not in existing_cuts: #
                    self.cut_pool.append(cut) #

    def get_cuts(self) -> list:
        return list(self.cut_pool) #

    def get_best_cost(self) -> float:
        return self.best_cost.value #

    def increment_idle_worker_count(self):
        with self.lock: self.idle_workers.value += 1 #

    def decrement_idle_worker_count(self):
        with self.lock: self.idle_workers.value -= 1 #

    def get_idle_worker_count(self) -> int:
        return self.idle_workers.value #

    def increment_nodes_processed(self):
        with self.lock: self.nodes_processed.value += 1 #