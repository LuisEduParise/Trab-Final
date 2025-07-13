# main.py
"""
Ponto de entrada principal para a execução do resolvedor de MIP.

Este script orquestra o processo completo:
1. Carrega um problema de MIP de um arquivo (formato .MPS ou customizado).
2. Executa a fase de pré-processamento (presolve) para simplificar o problema.
3. Inicia o resolvedor paralelo com a estratégia de work-stealing.
4. Exibe os resultados e estatísticas finais.
"""
import multiprocessing as mp
import time

from problem_parser import create_problem_from_mps
from problem_parser import create_problem_from_file
from presolver import Presolver, InfeasibleProblemError
from work_stealing_solver import WorkStealingSolver


def main():
    """
    Função principal que configura e executa as fases de resolução do problema.
    """
    main_start_time = time.time()

    try:
        # --- Seleção do Problema ---
        # Escolha uma das opções abaixo para carregar um problema.

        # Opção 1: Carregar de um arquivo no formato padrão .MPS
        problem = create_problem_from_mps("instances/fastxgemm-n2r6s0t2.mps")

        # Opção 2: Carregar de um arquivo de texto no formato customizado
        # problem = create_problem_from_file("instances/knapsack_test_2.txt")

        print(f"\nProblema '{problem.name}' carregado.")
        print(f"Sentido da Otimização: {problem.sense.upper()}")
        print(f"{len(problem.variables)} variáveis, {len(problem.constraints)} restrições.")
        print("="*50)
        
        # Fase 1: Pré-processamento (Presolve)
        print("\nIniciando a fase de Presolve...")
        presolver = Presolver(problem, use_probing=True)
        presolve_start_time = time.time()
        simplified_problem = presolver.presolve()
        presolve_time = time.time() - presolve_start_time
        
        # Fase 2: Resolução com Work Stealing
        solver = WorkStealingSolver(
            problem=simplified_problem,
            num_workers=None,  # Usa o número de CPUs como padrão
            timeout=600 - presolve_time,  # Timeout total de 10 minutos
            stagnation_limit=500000,      # Limite de nós sem melhoria
            mip_gap_tolerance=0.00001    # Gap de 0.001%
        )
        
        best_cost, best_solution = solver.solve()
        
        total_elapsed_time = time.time() - main_start_time

        # Exibição dos resultados
        if best_solution:
            print("\n--- Melhor Solução Encontrada ---")
            # Filtra e exibe apenas as variáveis com valor não-zero
            solution_vars = {k: v for k, v in best_solution.items() if abs(v) > 1e-6}
            for var_name, var_value in sorted(solution_vars.items()):
                print(f"  {var_name}: {var_value:.2f}")
            print("-----------------------------------")

        print("\n--- Estatísticas de Tempo ---")
        print(f"Tempo da fase de Presolve: {presolve_time:.4f} segundos")
        print(f"Tempo total de execução:   {total_elapsed_time:.4f} segundos")
        print("-----------------------------")

    except FileNotFoundError:
        print("\n!!! ERRO: O arquivo de problema especificado não foi encontrado.")
        print("Por favor, verifique o caminho do arquivo em main.py.")
    except InfeasibleProblemError as e:
        print(f"\nPROCESSO ENCERRADO PELO PRESOLVE.")
        print(f"RAZÃO: O problema foi provado como inviável. Detalhes: {e}")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # `freeze_support()` é necessário para criar executáveis em alguns sistemas
    # operacionais, como o Windows.
    mp.freeze_support()
    main()