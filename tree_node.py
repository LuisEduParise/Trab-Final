from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Node:
    # --- ALTERADO: Adicionado lp_bound para armazenar o valor real do LP pai ---
    lp_bound: Optional[float] = None
    
    # Atributo para ordenação do heap. Pode ser o lp_bound ou -lp_bound para maximização.
    parent_objective: Optional[float] = None
    
    extra_bounds: List[Tuple[str, str, float]] = field(default_factory=list)
    branch_variable: Optional[str] = None
    branch_direction: Optional[str] = None
    depth: int = 0
    
    strategy: str = "best_bound"

    def __lt__(self, other: 'Node') -> bool:
        # A lógica de comparação usa 'parent_objective' que é preparado para ordenação.
        if Node.strategy == "dfs":
            if self.depth != other.depth:
                return self.depth > other.depth # Maior profundidade primeiro
            # Se a profundidade for a mesma, usa o best-bound como critério de desempate
            if self.parent_objective is not None and other.parent_objective is not None:
                return self.parent_objective < other.parent_objective
            return False

        else: # "best_bound"
            if self.parent_objective is None: return False
            if other.parent_objective is None: return True
            
            # Compara o valor preparado para ordenação (menor é melhor)
            if self.parent_objective != other.parent_objective:
                return self.parent_objective < other.parent_objective
            
            # Desempate pela profundidade (maior profundidade primeiro)
            return self.depth > other.depth
