# implication_graph.py
"""
Define a estrutura de dados para representar um Grafo de Implicações.

Este grafo armazena as consequências lógicas de se fixar uma variável binária
em um determinado valor (0 ou 1). Especificamente, ele mapeia uma fixação 
(ex: 'x_A' = 1) para os novos limites (bounds) que essa fixação implica em 
outras variáveis do problema.

A estrutura é projetada para ser usada pela fase de pré-processamento (presolve)
para encontrar reduções de domínio e agregações de variáveis, conforme descrito
na tese de T. Achterberg, seção 10.7[cite: 2013].
"""
from collections import defaultdict
from typing import Dict, Tuple, Set

class ImplicationGraph:
    """
    Representa e gerencia um grafo de implicações para um problema de MIP.

    O grafo é armazenado como um dicionário aninhado. A estrutura é:
    {
        'nome_var_binaria': {
            0: {'var_implicada': (lb, ub), ...},
            1: {'var_implicada': (lb, ub), ...}
        }, ...
    }
    Onde (lb, ub) são os limites inferior e superior implicados na 'var_implicada'.
    Usa-se float('-inf') e float('inf') para representar a ausência de um limite.
    """

    def __init__(self):
        """
        Inicializa um grafo de implicações vazio.
        A estrutura aninhada é criada sob demanda usando defaultdict.
        """
        # Formato: {bin_var: {bin_val: {implied_var: (lb, ub)}}}
        self.graph: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = \
            defaultdict(lambda: {0: {}, 1: {}})

    def add_implication(self, binary_var_name: str, binary_val: int, 
                        implied_var_name: str, bound_type: str, bound_val: float):
        """
        Adiciona ou atualiza uma implicação no grafo.

        Se uma implicação para a mesma variável já existe, ela é atualizada apenas
        se o novo limite for mais restritivo.

        Args:
            binary_var_name (str): O nome da variável binária cuja fixação causa a implicação.
            binary_val (int): O valor (0 ou 1) da variável binária.
            implied_var_name (str): O nome da variável que sofre a implicação.
            bound_type (str): O tipo de limite implicado ('lb' ou 'ub').
            bound_val (float): O valor do limite implicado.
        """
        implications = self.graph[binary_var_name][binary_val]
        
        current_lb, current_ub = implications.get(implied_var_name, (float('-inf'), float('inf')))
        
        made_change = False
        if bound_type == 'lb':
            if bound_val > current_lb:
                implications[implied_var_name] = (bound_val, current_ub)
                made_change = True
        elif bound_type == 'ub':
            if bound_val < current_ub:
                implications[implied_var_name] = (current_lb, bound_val)
                made_change = True
        
        return made_change

    def get_implied_bounds(self, binary_var_name: str, binary_val: int, implied_var_name: str) -> Tuple[float, float]:
        """
        Retorna os limites implicados para uma variável, dado um gatilho binário.

        Returns:
            Tuple[float, float]: Uma tupla (lb, ub) com os limites. Retorna 
                                 (-inf, inf) se não houver implicação.
        """
        return self.graph[binary_var_name][binary_val].get(implied_var_name, (float('-inf'), float('inf')))

    def get_common_implied_vars(self, binary_var_name: str) -> Set[str]:
        """
        Encontra o conjunto de variáveis que sofrem implicações tanto para a fixação
        da variável binária em 0 quanto em 1.

        Este é o passo chave para a análise do grafo, pois é nessas variáveis que
        podemos deduzir novos limites globais ou agregações.

        Returns:
            Set[str]: Um conjunto com os nomes das variáveis implicadas em ambos os casos.
        """
        vars_on_0 = set(self.graph[binary_var_name][0].keys())
        vars_on_1 = set(self.graph[binary_var_name][1].keys())
        
        return vars_on_0.intersection(vars_on_1)