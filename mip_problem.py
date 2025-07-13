# mip_problem.py
"""
Define as estruturas de dados fundamentais para representar um Problema de
Programação Inteira Mista (MIP).

Este módulo contém as dataclasses que modelam as variáveis, as restrições e
o problema como um todo, servindo como a base para os módulos de análise e
resolução.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import copy


@dataclass
class Variable:
    """
    Representa uma variável de decisão do problema de otimização.

    Attributes:
        name (str): O nome único da variável (ex: 'x_1', 'y_fabrica_A').
        is_integer (bool): True se a variável deve assumir valores inteiros,
                           False caso contrário. O padrão é False.
        lb (float): O limite inferior (lower bound) da variável. Padrão é 0.0.
        ub (float): O limite superior (upper bound) da variável. Padrão é infinito.
    """
    name: str
    is_integer: bool = False
    lb: float = 0.0
    ub: float = float('inf')


@dataclass
class Constraint:
    """
    Representa uma restrição linear do problema.

    A restrição é modelada no formato: a*x + b*y (sense) c, onde 'sense'
    pode ser '<=', '>=', ou '=='.

    Attributes:
        coeffs (Dict[str, float]): Dicionário mapeando nomes de variáveis
                                   aos seus respectivos coeficientes na expressão.
        sense (str): O sentido da restrição ('<=', '>=', '==').
        rhs (float): O valor do lado direito (right-hand side) da restrição.
    """
    coeffs: Dict[str, float]
    sense: str
    rhs: float

    def __hash__(self):
        """
        Calcula o hash da restrição para permitir sua utilização em coleções
        como conjuntos (sets) e como chaves de dicionários.
        """
        return hash((frozenset(self.coeffs.items()), self.sense, self.rhs))

    def __eq__(self, other):
        """
        Verifica a igualdade entre duas restrições.
        """
        if not isinstance(other, Constraint):
            return NotImplemented
        return self.coeffs == other.coeffs and self.sense == other.sense and self.rhs == other.rhs


@dataclass
class MIPProblem:
    """
    Representa a definição completa de um Problema de Programação Inteira Mista.

    Esta classe agrega todos os componentes de um problema de otimização.

    Attributes:
        name (str): O nome do problema.
        variables (List[Variable]): Uma lista de objetos Variable.
        objective (Dict[str, float]): Dicionário que define a função objetivo,
                                      mapeando nomes de variáveis a coeficientes.
        constraints (List[Constraint]): Uma lista de objetos Constraint.
        sense (str): O sentido da otimização ("minimize" ou "maximize").
    """
    name: str
    variables: List[Variable]
    objective: Dict[str, float]
    constraints: List[Constraint]
    sense: str = "minimize"

    def copy(self):
        """
        Cria e retorna uma cópia profunda (deep copy) do objeto MIPProblem.

        Returns:
            MIPProblem: Uma instância completamente nova e independente do problema.
        """
        return copy.deepcopy(self)