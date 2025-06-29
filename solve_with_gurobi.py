# solve_with_gurobi.py
import gurobipy as gp
from gurobipy import GRB

# Importamos nossa definição de problema e o criador do problema complexo
from mip_problem import MIPProblem
from factory_location_problem import create_factory_location_problem

def solve_mip_with_gurobi(problem: MIPProblem):
    """
    Resolve um MIPProblem usando o solver Gurobi.
    Esta função atua como um 'tradutor' da nossa estrutura agnóstica
    para o formato específico do Gurobi.
    """
    try:
        # --- 1. Inicialização do Modelo Gurobi ---
        model = gp.Model(problem.name)

        # --- 2. Tradução das Variáveis ---
        # Criamos um dicionário para mapear nomes de variáveis para objetos Gurobi
        gurobi_vars = {}
        for var_spec in problem.variables:
            var_type = GRB.BINARY if var_spec.is_integer else GRB.CONTINUOUS
            gurobi_vars[var_spec.name] = model.addVar(name=var_spec.name, vtype=var_type)

        # --- 3. Tradução da Função Objetivo ---
        objective_expr = gp.LinExpr()
        for var_name, coeff in problem.objective.items():
            if var_name in gurobi_vars:
                objective_expr += coeff * gurobi_vars[var_name]

        model_sense = GRB.MINIMIZE if problem.sense == "minimize" else GRB.MAXIMIZE
        model.setObjective(objective_expr, model_sense)

        # --- 4. Tradução das Restrições (BLOCO CORRIGIDO) ---
        for i, const_spec in enumerate(problem.constraints):
            lhs_expr = gp.LinExpr()
            for var_name, coeff in const_spec.coeffs.items():
                if var_name in gurobi_vars:
                    lhs_expr += coeff * gurobi_vars[var_name]
            
            # Adiciona a restrição usando sobrecarga de operadores (forma recomendada)
            if const_spec.sense == "<=":
                model.addConstr(lhs_expr <= const_spec.rhs, name=f"C{i}")
            elif const_spec.sense == ">=":
                model.addConstr(lhs_expr >= const_spec.rhs, name=f"C{i}")
            elif const_spec.sense == "==":
                model.addConstr(lhs_expr == const_spec.rhs, name=f"C{i}")
            else:
                raise ValueError(f"Sentido da restrição '{const_spec.sense}' não é suportado.")

        # --- 5. Resolução e Apresentação dos Resultados ---
        print("--- Resolvendo o problema com Gurobi ---")
        model.optimize()

        # Verifica o status da solução
        if model.Status == GRB.OPTIMAL:
            print("\n--- Solução Ótima Encontrada ---")
            print(f"Valor Objetivo (Custo Total Mínimo): ${model.ObjVal:,.2f}")

            print("\nDecisões:")
            for var in model.getVars():
                if var.VarName.startswith("open_") and var.X > 0.5:
                    print(f"  - [ABRIR] Fábrica {var.VarName.split('_')[1]}")
                elif var.VarName.startswith("ship_") and var.X > 1e-6:
                    _, f, c = var.VarName.split("_")
                    print(f"  - [ENVIAR] {var.X:.1f} unidades da Fábrica {f} para o Cliente {c}")
        
        elif model.Status == GRB.INFEASIBLE:
            print("\nO problema é inviável. Não existe solução que satisfaça todas as restrições.")
            model.computeIIS()
            model.write("inviavel.ilp")
            print("Um arquivo 'inviavel.ilp' foi gerado para ajudar a identificar a causa.")

        else:
            print(f"\nOtimização finalizada com status: {model.Status}")

    except gp.GurobiError as e:
        print(f"Erro do Gurobi: {e.errno} - {e}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")


if __name__ == '__main__':
    # 1. Cria a instância do problema GRANDE usando o gerador aprimorado
    #    O uso da 'seed' garante que o problema seja sempre o mesmo.
    large_factory_problem = create_factory_location_problem(
        num_fabricas=200, 
        num_clientes=800, 
        seed=123
    )
    
    # 2. Resolve o problema usando Gurobi
    solve_mip_with_gurobi(large_factory_problem)