# factory_location_problem.py
import random
from mip_problem import Variable, Constraint, MIPProblem
from typing import List, Dict

def create_factory_location_problem(
    num_fabricas: int = 3, 
    num_clientes: int = 4, 
    seed: int = None
) -> MIPProblem:
    """
    Cria um Problema de Localização de Fábricas Capacitado, agora de tamanho configurável.

    Argumentos:
        num_fabricas (int): O número de locais de fábricas a serem considerados.
        num_clientes (int): O número de clientes a serem atendidos.
        seed (int, optional): Uma semente para o gerador de números aleatórios para
                              garantir a reprodutibilidade do problema.
    """
    if seed is not None:
        random.seed(seed)

    print(f"\nGerando um problema com {num_fabricas} fábricas e {num_clientes} clientes...")

    # --- 1. Geração Programática dos Parâmetros ---
    
    fabricas = [f"F{i+1}" for i in range(num_fabricas)]
    clientes = [f"C{i+1}" for i in range(num_clientes)]

    # Gera custos, capacidades e demandas aleatoriamente dentro de faixas
    custos_fixos = {f: random.randint(800, 2000) for f in fabricas}
    capacidades = {f: random.randint(500, 1500) for f in fabricas}
    demandas = {c: random.randint(50, 200) for c in clientes}
    
    # Gera custos de transporte para cada rota (fábrica -> cliente)
    custos_transporte = {
        f: {c: random.randint(2, 10) for c in clientes} for f in fabricas
    }

    # --- 2. Garantia de Viabilidade ---
    # Garante que a capacidade total de todas as fábricas seja suficiente para
    # atender à demanda total. Isso evita a criação de problemas inviáveis.
    total_demand = sum(demandas.values())
    total_capacity = sum(capacidades.values())

    # Asseguramos que a capacidade total seja pelo menos 40% maior que a demanda total
    if total_capacity < total_demand * 1.4:
        # Se não for, calculamos o fator de escala necessário e ajustamos as capacidades
        scale_factor = (total_demand * 1.4) / total_capacity
        for f in fabricas:
            capacidades[f] = int(capacidades[f] * scale_factor)
        print("Aviso: As capacidades foram aumentadas para garantir a viabilidade do problema.")


    # --- 3. Criação das Variáveis e do Objetivo (lógica idêntica à anterior) ---
    
    variables: List[Variable] = []
    objective: Dict[str, float] = {}

    # Variáveis binárias 'open_f'
    for f in fabricas:
        var_name = f"open_{f}"
        variables.append(Variable(name=var_name, is_integer=True))
        objective[var_name] = custos_fixos[f]

    # Variáveis contínuas 'ship_f_c'
    for f in fabricas:
        for c in clientes:
            var_name = f"ship_{f}_{c}"
            variables.append(Variable(name=var_name, is_integer=False))
            objective[var_name] = custos_transporte[f][c]

    # --- 4. Criação das Restrições (lógica idêntica à anterior) ---

    constraints: List[Constraint] = []

    # Restrições de Demanda
    for c in clientes:
        coeffs = {f"ship_{f}_{c}": 1.0 for f in fabricas}
        constraints.append(Constraint(coeffs=coeffs, sense="==", rhs=demandas[c]))

    # Restrições de Capacidade (ligação)
    for f in fabricas:
        coeffs = {f"ship_{f}_{c}": 1.0 for c in clientes}
        coeffs[f"open_{f}"] = -capacidades[f]
        constraints.append(Constraint(coeffs=coeffs, sense="<=", rhs=0))
        
    # Restrições de binariedade (open <= 1)
    for f in fabricas:
        constraints.append(Constraint(coeffs={f"open_{f}": 1.0}, sense="<=", rhs=1))


    # --- 5. Montagem Final do Problema ---
    
    problem_name = f"FactoryLocation_{num_fabricas}f_{num_clientes}c"
    factory_problem = MIPProblem(
        name=problem_name,
        variables=variables,
        objective=objective,
        constraints=constraints,
        sense="minimize"
    )

    return factory_problem


# --- Bloco de Exemplo de Uso ---
if __name__ == '__main__':
    # Exemplo 1: Gerar o mesmo problema pequeno de antes para consistência
    small_problem = create_factory_location_problem(3, 4, seed=42)
    print(f"--- Problema Pequeno Gerado: {small_problem.name} ---")
    print(f"Total de Variáveis: {len(small_problem.variables)}")
    print(f"Total de Restrições: {len(small_problem.constraints)}")

    # Exemplo 2: Gerar um problema realmente grande para testar o solver
    # 50 fábricas e 200 clientes resultarão em:
    # - Variáveis: 50 (open) + 50 * 200 (ship) = 10.050
    # - Restrições: 200 (demanda) + 50 (capacidade) + 50 (binárias) = 300
    large_problem = create_factory_location_problem(50, 200, seed=123)
    print(f"\n--- Problema Grande Gerado: {large_problem.name} ---")
    print(f"Total de Variáveis: {len(large_problem.variables)}")
    print(f"Total de Restrições: {len(large_problem.constraints)}")