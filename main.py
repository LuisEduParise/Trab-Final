# main.py
import multiprocessing as mp
import time

# --- ALTERAÇÃO ---
# Importe a nova função do parser
from problem_parser import create_problem_from_mps
from problem_parser import create_problem_from_file
# A linha abaixo pode ser mantida se você ainda quiser gerar problemas programaticamente
from factory_location_problem import create_factory_location_problem

from presolver import Presolver, InfeasibleProblemError
from work_stealing_solver import WorkStealingSolver


def main():
    """
    Ponto de entrada principal para configurar e resolver o problema de MIP.
    """
    main_start_time = time.time()

    # --- ALTERAÇÃO ---
    # Agora você pode carregar um problema de um arquivo .MPS.
    # Descomente a linha abaixo e substitua pelo caminho do seu arquivo.
    # Exemplo: problem = create_problem_from_mps("data/problemas/meu_problema.mps")
    
    try:
        # Opção 1: Carregar de um arquivo .MPS
        problem = create_problem_from_mps("instances/mas76.mps")

        # Opção 2: Gerar um problema programaticamente (mantenha comentado se usar MPS)
        #problem = create_factory_location_problem(
        #     num_fabricas=50, 
        #     num_clientes=200, 
        #    seed=123
        #)

        # Opção 3: João Knapsack
        #problem = create_problem_from_file("/instances/knapsack_test_2.txt")

        print(f"\nProblema '{problem.name}' carregado. {len(problem.variables)} variáveis, {len(problem.constraints)} restrições.")
        print("\n" + "="*50)
        print(f"Problema '{problem.name}' carregado.")
        print(f"Sentido da Otimização: {problem.sense.upper()}")
        print(f"{len(problem.variables)} variáveis, {len(problem.constraints)} restrições.")
        print("="*50)
        
        print("\nIniciando a fase de Presolve...")
        presolver = Presolver(problem, use_probing=True)
        
        presolve_start_time = time.time()
        simplified_problem = presolver.presolve()
        presolve_time = time.time() - presolve_start_time
        
        solver = WorkStealingSolver(
            simplified_problem,
            num_workers=None,
            timeout=600-presolve_time,
            stagnation_limit=500000,
            mip_gap_tolerance=0.00001
        )
        
        best_cost, best_solution = solver.solve()
        
        total_elapsed_time = time.time() - main_start_time

        if best_solution:
            print("\n--- Melhor Solução Encontrada ---")
            solution_vars = {k: v for k, v in best_solution.items() if abs(v) > 1e-6}
            for var_name, var_value in sorted(solution_vars.items()):
                print(f"  {var_name}: {var_value:.2f}")
            print("-----------------------------------")

        print("\n--- Estatísticas de Tempo ---")
        print(f"Tempo da fase de Presolve: {presolve_time:.4f} segundos")
        print(f"Tempo total de execução:   {total_elapsed_time:.4f} segundos")
        print("-----------------------------")

    except FileNotFoundError:
        print("\n!!! ERRO: O arquivo .MPS especificado não foi encontrado.")
        print("Por favor, verifique o caminho do arquivo em main.py.")
    except InfeasibleProblemError as e:
        print(f"\nO PROCESSO FOI ENCERRADO PELO PRESOLVE.")
        print(f"RAZÃO: O problema foi provado como inviável. Detalhes: {e}")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    mp.freeze_support()
    main()