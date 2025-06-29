# solve_with_gurobi.py
import gurobipy as gp
from gurobipy import GRB

from mip_problem import MIPProblem
from problem_parser import create_problem_from_file

def main():
    """
    Carrega um problema usando o parser e o resolve com Gurobi para validação.
    """
    print("--- Iniciando Validação com Gurobi ---")

    # 1. Carregar o problema usando o parser que criamos
    try:
        problem = create_problem_from_file('knapsack_test_2.txt')
        print(f"Problema '{problem.name}' carregado com sucesso pelo parser.")
        print(f"Otimizando para: {problem.sense}")
    except Exception as e:
        print(f"Falha ao carregar o problema com o parser: {e}")
        return

    # 2. Construir o modelo Gurobi a partir do objeto MIPProblem
    try:
        print("\nConstruindo o modelo Gurobi...")
        model = gp.Model(problem.name)

        # Adicionar variáveis ao modelo Gurobi
        gurobi_vars = {}
        for var in problem.variables:
            # Determina o tipo da variável para o Gurobi
            vtype = GRB.CONTINUOUS
            if var.is_integer:
                # O parser define variáveis binárias com lb=0 e ub=1
                vtype = GRB.BINARY if var.lb == 0 and var.ub == 1 else GRB.INTEGER
            
            gurobi_vars[var.name] = model.addVar(
                lb=var.lb, ub=var.ub, vtype=vtype, name=var.name
            )

        # Definir o sentido da otimização
        if problem.sense == "maximize":
            model.ModelSense = GRB.MAXIMIZE
        else:
            model.ModelSense = GRB.MINIMIZE

        # Definir a função objetivo
        objective_expr = gp.LinExpr()
        for var_name, coeff in problem.objective.items():
            objective_expr += coeff * gurobi_vars[var_name]
        model.setObjective(objective_expr)

        # Adicionar as restrições
        for const in problem.constraints:
            lhs_expr = gp.LinExpr()
            for var_name, coeff in const.coeffs.items():
                lhs_expr += coeff * gurobi_vars[var_name]
            
            if const.sense == "<=":
                model.addConstr(lhs_expr <= const.rhs)
            elif const.sense == ">=":
                model.addConstr(lhs_expr >= const.rhs)
            else: # "=="
                model.addConstr(lhs_expr == const.rhs)
        
        print("Modelo Gurobi construído com sucesso.")

    except Exception as e:
        print(f"Falha ao construir o modelo Gurobi: {e}")
        return

    # 3. Otimizar o modelo
    print("\n--- Otimizando com Gurobi ---")
    model.optimize()

    # 4. Exibir os resultados
    print("\n--- Resultados do Gurobi ---")
    if model.Status == GRB.OPTIMAL:
        print(f"Solução ótima encontrada!")
        print(f"Valor da Função Objetivo: {model.ObjVal:.4f}")
        
        print("\nVariáveis selecionadas (com valor > 0):")
        for v in model.getVars():
            if v.X > 1e-6: # Usar uma tolerância pequena para evitar ruído numérico
                print(f"  - {v.VarName} = {v.X:.0f}")
    
    elif model.Status == GRB.INFEASIBLE:
        print("O problema é inviável (infeasible).")
        print("Isso indica que não existe solução que satisfaça todas as restrições.")
        print("Verifique as restrições do seu problema.")

    elif model.Status == GRB.UNBOUNDED:
        print("O problema é ilimitado (unbounded).")
        print("A função objetivo pode crescer (ou diminuir) infinitamente.")

    else:
        print(f"Otimização encerrada com status: {model.Status}")

if __name__ == "__main__":
    # Certifique-se de que a biblioteca gurobipy está instalada:
    # pip install gurobipy
    main()