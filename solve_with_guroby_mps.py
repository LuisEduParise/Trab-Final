# solve_with_gurobi.py
import gurobipy as gp
from gurobipy import GRB
import sys

def solve_mps_with_gurobi(filepath: str):
    """
    Lê e resolve um arquivo MPS usando o solver Gurobi padrão.

    Args:
        filepath: O caminho para o arquivo .mps.
    """
    try:
        # Cria um ambiente Gurobi
        env = gp.Env(empty=True)
        # Desativa a saída de licença no console para um log mais limpo
        env.setParam('LogToConsole', 0)
        env.start()

        # Lê o modelo do arquivo. Gurobi lida com todo o parsing.
        print(f"\n--- Verificador Gurobi: Lendo o arquivo '{filepath}' ---")
        model = gp.read(filepath, env=env)
        
        # Ativa o log do Gurobi no console para vermos o progresso
        model.setParam('LogToConsole', 1)
        
        # Otimiza o modelo
        print("\n--- Verificador Gurobi: Iniciando a otimização... ---")
        model.optimize()

        # Imprime os resultados
        print("\n" + "="*50)
        print("--- Verificador Gurobi: Resultados Finais ---")
        print(f"Status da Otimização: {model.Status} ({GRB.Status.OPTIMAL} = Ótimo)")

        if model.Status == GRB.OPTIMAL:
            print(f"Sentido da Otimização: {'MINIMIZE' if model.ModelSense == GRB.MINIMIZE else 'MAXIMIZE'}")
            print(f"Valor da Função Objetivo: {model.ObjVal:.4f}")
            
            print("\n--- Solução (variáveis não nulas) ---")
            solution_found = False
            for v in model.getVars():
                if abs(v.X) > 1e-6:
                    print(f"  {v.VarName}: {v.X:.2f}")
                    solution_found = True
            if not solution_found:
                print("  Nenhuma variável com valor não-nulo na solução.")

        elif model.Status == GRB.INFEASIBLE:
            print("O modelo foi provado como inviável.")
        
        elif model.Status == GRB.UNBOUNDED:
            print("O modelo foi provado como ilimitado (unbounded).")

        else:
            print("Nenhuma solução ótima foi encontrada.")
        
        print("="*50)

    except gp.GurobiError as e:
        print(f"Erro do Gurobi: {e.errno} - {e}")
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{filepath}'")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")

if __name__ == "__main__":
    # Verifica se o caminho do arquivo foi passado como argumento
    if len(sys.argv) > 1:
        file_to_solve = sys.argv[1]
    else:
        # Se nenhum argumento for passado, usa um nome padrão.
        # Altere este nome para o seu arquivo .mps.
        file_to_solve = "instances/instance_0003.mps"
        print(f"AVISO: Nenhum arquivo especificado. Usando o padrão: '{file_to_solve}'")
        print("Uso: python solve_with_gurobi.py <caminho_para_o_arquivo.mps>")
    
    solve_mps_with_gurobi(file_to_solve)
