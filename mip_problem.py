# mip_problem.py
from dataclasses import dataclass, field
from typing import List, Dict
import copy

@dataclass
class Variable:
    """Define uma variável do problema, incluindo seus bounds e tipo."""
    name: str
    is_integer: bool = False
    lb: float = 0.0
    ub: float = float('inf')

@dataclass
class Constraint:
    """Define uma restrição linear no formato: a*x + b*y <= c."""
    coeffs: Dict[str, float]
    sense: str
    rhs: float

    # Permite que a restrição seja usada em conjuntos e como chave de dicionário
    def __hash__(self):
        return hash((frozenset(self.coeffs.items()), self.sense, self.rhs))

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return NotImplemented
        return self.coeffs == other.coeffs and self.sense == other.sense and self.rhs == other.rhs

@dataclass
class MIPProblem:
    """Representa a definição completa de um Problema de Programação Inteira Mista."""
    name: str
    variables: List[Variable]
    objective: Dict[str, float]
    constraints: List[Constraint]
    sense: str = "minimize"

    def copy(self):
        """Retorna uma cópia profunda (deep copy) do problema."""
        return copy.deepcopy(self)