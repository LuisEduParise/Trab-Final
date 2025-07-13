# tree_node.py
"""
Define a estrutura de dados para um nó na árvore de busca do algoritmo
Branch and Bound.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Node:
    """
    Representa um nó na árvore de busca do Branch and Bound.

    Cada nó corresponde a um subproblema do problema original, definido pela
    adição de restrições de limite (bounds) às variáveis de ramificação.

    Attributes:
        lp_bound (Optional[float]): O valor da função objetivo da relaxação linear
                                    resolvida no nó pai. Usado para cálculos
                                    como o de pseudocustos.
        parent_objective (Optional[float]): O valor usado para ordenar o nó na
                                            fila de prioridade (heap). Para
                                            minimização, é o próprio lp_bound.
                                            Para maximização, seria -lp_bound.
        extra_bounds (List[Tuple[str, str, float]]): Lista de tuplas que definem
                                                     os limites adicionais das
                                                     variáveis para este subproblema.
                                                     Ex: [('x_1', '<=', 0.0)].
        branch_variable (Optional[str]): O nome da variável na qual o branching
                                         ocorreu para criar este nó.
        branch_direction (Optional[str]): A direção do branching ('up' ou 'down').
        depth (int): A profundidade do nó na árvore de busca (nó raiz tem profundidade 0).
        strategy (str): Um atributo de classe que controla a estratégia de
                        ordenação para todos os nós. Pode ser "dfs" (Depth First Search)
                        ou "best_bound".
    """
    lp_bound: Optional[float] = None
    parent_objective: Optional[float] = None
    extra_bounds: List[Tuple[str, str, float]] = field(default_factory=list)
    branch_variable: Optional[str] = None
    branch_direction: Optional[str] = None
    depth: int = 0
    
    strategy: str = "best_bound"

    def __lt__(self, other: 'Node') -> bool:
        """
        Define a lógica de comparação para ordenar os nós em uma fila de prioridade (heap).

        A ordenação depende da estratégia de busca definida no atributo de classe `strategy`:
        - "dfs": Prioriza nós com maior profundidade. Em caso de empate, usa o
                 critério de best-bound.
        - "best_bound": Prioriza nós com o melhor valor de `parent_objective`.
                        Em caso de empate, prioriza o de maior profundidade.

        Args:
            other (Node): O outro nó com o qual este será comparado.

        Returns:
            bool: True se este nó deve ter prioridade sobre o outro.
        """
        # Estratégia de busca em profundidade (Depth-First Search)
        if Node.strategy == "dfs":
            if self.depth != other.depth:
                return self.depth > other.depth  # Maior profundidade tem maior prioridade
            # Critério de desempate: melhor bound
            if self.parent_objective is not None and other.parent_objective is not None:
                return self.parent_objective < other.parent_objective
            return False

        # Estratégia de busca por melhor limite (Best-Bound)
        else:
            if self.parent_objective is None: return False
            if other.parent_objective is None: return True
            
            # Compara o valor do objetivo (menor é melhor para um min-heap)
            if self.parent_objective != other.parent_objective:
                return self.parent_objective < other.parent_objective
            
            # Critério de desempate: maior profundidade
            return self.depth > other.depth