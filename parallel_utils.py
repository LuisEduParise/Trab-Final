# parallel_utils.py
"""
Fornece utilitários para gerenciar o estado compartilhado em um ambiente
de multiprocessamento.

A classe `SharedState` é o componente central, garantindo que os múltiplos
processos 'workers' possam acessar e modificar dados globais (como a melhor
solução encontrada) de forma segura e sincronizada.
"""
import multiprocessing as mp
from mip_problem import Constraint
from typing import Dict, List, Optional


class SharedState:
    """
    Gerencia o estado global compartilhado entre todos os processos workers.

    Utiliza um `multiprocessing.Manager` para criar objetos que podem ser
    compartilhados de forma segura entre processos, como valores, dicionários
    e listas. Locks são usados para prevenir condições de corrida durante
    operações de atualização.
    """
    def __init__(self, num_workers: int, sense: str = "minimize"):
        """
        Inicializa o estado compartilhado.

        Args:
            num_workers (int): O número de processos worker que compartilharão este estado.
            sense (str): O sentido da otimização ("minimize" ou "maximize").
        """
        self.num_workers = num_workers
        self.sense = sense
        
        manager = mp.Manager()
        
        # O valor inicial da melhor solução depende do sentido da otimização
        initial_primal = float('inf') if self.sense == 'minimize' else -float('inf')
        
        # O melhor custo (valor da função objetivo) da solução incumbente
        self.best_cost = mp.Value('d', initial_primal)
        
        # A melhor solução (valores das variáveis) encontrada até agora
        self.best_solution = manager.dict()
        
        # Flag que indica se alguma solução viável já foi encontrada
        self.has_solution = mp.Value('b', False)
        
        # Contador de workers que estão atualmente ociosos (sem nós para processar)
        self.idle_workers = mp.Value('i', 0)
        
        # Contador total de nós da árvore de B&B processados por todos os workers
        self.nodes_processed = mp.Value('i', 0)
        
        # Armazena o `nodes_processed` no momento da última melhoria da solução
        self.last_update_node_count = mp.Value('i', 0)
        
        # Dicionário que armazena o melhor bound local de cada worker
        self.worker_best_bounds = manager.dict({i: initial_primal for i in range(num_workers)})
        
        # Lista compartilhada de cortes (cutting planes) globais
        self.cut_pool = manager.list()
        
        # Locks para garantir a atomicidade das operações
        self.cut_lock = mp.Lock()
        self.lock = mp.Lock()

    def update_best_solution(self, cost: float, solution: Dict[str, float]) -> bool:
        """
        Atualiza a melhor solução global se a nova solução for melhor.

        Esta operação é atômica (thread-safe) graças ao uso de um lock.

        Args:
            cost (float): O custo (valor da função objetivo) da nova solução candidata.
            solution (dict): Um dicionário com os valores das variáveis da nova solução.

        Returns:
            bool: True se a solução global foi atualizada, False caso contrário.
        """
        with self.lock:
            is_better = False
            if self.sense == 'minimize':
                if cost < self.best_cost.value:
                    is_better = True
            else:  # maximize
                if cost > self.best_cost.value:
                    is_better = True
            
            if is_better:
                self.best_cost.value = cost
                
                # Atualiza o dicionário da solução de forma segura
                self.best_solution.clear()
                self.best_solution.update(solution)
                
                if not self.has_solution.value:
                    self.has_solution.value = True
                
                self.last_update_node_count.value = self.nodes_processed.value
                return True
        return False

    def get_best_solution(self) -> Dict[str, float]:
        """
        Retorna uma cópia da melhor solução encontrada de forma segura.

        Returns:
            Dict[str, float]: Um dicionário representando a melhor solução, ou um
                              dicionário vazio se nenhuma solução foi encontrada.
        """
        with self.lock:
            if self.has_solution.value:
                return dict(self.best_solution)
            return {}

    def update_worker_best_bound(self, worker_id: int, bound: float):
        """
        Atualiza o melhor bound local para um worker específico.

        Este valor representa o nó mais promissor na fila local do worker e é usado
        para calcular o dual bound global.

        Args:
            worker_id (int): O ID do worker que está reportando seu bound.
            bound (float): O valor do bound.
        """
        self.worker_best_bounds[worker_id] = bound
        
    def get_last_update_node(self) -> int:
        """
        Retorna o número total de nós processados no momento da última atualização
        da melhor solução.
        """
        return self.last_update_node_count.value

    def add_cuts(self, new_cuts: List[Constraint]):
        """
        Adiciona novos cortes (cutting planes) ao pool global de forma segura.

        Evita a adição de cortes duplicados.

        Args:
            new_cuts (List[Constraint]): Uma lista de novos cortes a serem adicionados.
        """
        with self.cut_lock:
            existing_cuts = set(self.cut_pool)
            for cut in new_cuts:
                if cut not in existing_cuts:
                    self.cut_pool.append(cut)

    def get_cuts(self) -> List[Constraint]:
        """Retorna uma cópia da lista de cortes globais."""
        return list(self.cut_pool)

    def get_best_cost(self) -> float:
        """Retorna o custo da melhor solução encontrada."""
        return self.best_cost.value

    def increment_idle_worker_count(self):
        """Incrementa o contador de workers ociosos de forma segura."""
        with self.lock:
            self.idle_workers.value += 1

    def decrement_idle_worker_count(self):
        """Decrementa o contador de workers ociosos de forma segura."""
        with self.lock:
            self.idle_workers.value -= 1

    def get_idle_worker_count(self) -> int:
        """Retorna o número atual de workers ociosos."""
        return self.idle_workers.value

    def increment_nodes_processed(self):
        """Incrementa o contador total de nós processados de forma segura."""
        with self.lock:
            self.nodes_processed.value += 1