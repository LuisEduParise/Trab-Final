# test_feasibility.py
import sys
from problem_parser import create_problem_from_mps
from work_stealing_solver import NodeProcessor

# Adiciona o diretório atual ao path para garantir que as importações funcionem
# caso o script seja executado de uma subpasta.
sys.path.append('.')


def test_feasibility_check():
    """
    Este script isola e testa a função _is_solution_feasible para verificar
    se ela consegue validar corretamente uma solução ótima já conhecida.
    """
    print("--- Iniciando Teste da Função _is_solution_feasible ---")

    # Dicionário com a solução ótima encontrada pelo Gurobi.
    # Variáveis não listadas aqui são consideradas como 0.0 pelo método .get() na verificação.
    solucao_conhecida = {
        "t": 12.0,
        "x6": 1.0,
        "x11": 1.0,
        "x12": 1.0,
        "x21": 1.0,
        "x24": 1.0,
        "x31": 1.0,
        "x59": 1.0,
        "x65": 1.0,
        "x74": 1.0,
        "x90": 1.0,
        "x147": 1.0,
        "x152": 1.0,
        "x154": 1.0,
        "Avar": 4.0,
        "A_hat_3": 4.0,
        "t_hat_3": 12.0,
        "w": 48.0,
        "z_3": 1.0,
    }
    print("Solução ótima conhecida foi carregada para o teste.")

    # Carrega o mesmo problema que o solver principal está usando
    try:
        problem = create_problem_from_mps("instances/instance_0003.mps")
        print(f"Problema '{problem.name}' carregado com sucesso a partir de 'instances/instance_0003.mps'.")
    except FileNotFoundError:
        print("\nERRO CRÍTICO: Arquivo 'instances/instance_0003.mps' não encontrado.")
        print("Por favor, certifique-se de que o script está na pasta raiz do projeto ou ajuste o caminho do arquivo.")
        return
    except Exception as e:
        print(f"\nOcorreu um erro inesperado ao carregar o problema: {e}")
        return


    # Instancia o NodeProcessor, que contém a função que queremos testar
    node_processor = NodeProcessor(problem)
    print("NodeProcessor instanciado.")

    print("\nTestando a viabilidade da solução conhecida...")

    # Chama a função que suspeitamos ter um bug
    is_feasible = node_processor._is_solution_feasible(solucao_conhecida)

    print("-" * 50)
    if is_feasible:
        print("RESULTADO: SUCESSO!")
        print("A função _is_solution_feasible considerou a solução ótima como VIÁVEL.")
        print("Isto sugere que o bug não está na verificação de viabilidade, mas sim na exploração da árvore de busca.")
    else:
        print("RESULTADO: FALHA!")
        print("A função _is_solution_feasible considerou a solução ótima como INVIÁVEL.")
        print("Isto confirma um BUG em _is_solution_feasible. Analisando qual restrição falhou...")

        # Se falhou, vamos descobrir exatamente onde para podermos corrigir
        for i, const in enumerate(node_processor.problem.constraints):
            # Recalcula a atividade da restrição para diagnóstico
            activity = sum(coeff * solucao_conhecida.get(var, 0) for var, coeff in const.coeffs.items())

            violated = False
            # Replica a mesma lógica de verificação da função original
            if const.sense == '<=' and activity > const.rhs + node_processor.tolerance:
                violated = True
            elif const.sense == '>=' and activity < const.rhs - node_processor.tolerance:
                violated = True
            elif const.sense == '==' and abs(activity - const.rhs) > node_processor.tolerance:
                violated = True

            if violated:
                print(f"\n!!! RESTRIÇÃO VIOLADA (Índice da restrição no problema: {i}) !!!")
                print(f"  - Definição da Restrição: ... {const.sense} {const.rhs}")
                print(f"  - Coeficientes Envolvidos: {const.coeffs}")
                print(f"  - Atividade Calculada (soma dos termos): {activity}")
                print(f"  - Detalhe da falha: O valor calculado '{activity}' não satisfez a condição '{const.sense} {const.rhs}' (com tolerância de {node_processor.tolerance})")
                break # Para no primeiro erro encontrado
        else:
            # Este bloco 'else' do for só é executado se o loop terminar sem 'break'
            print("\nAnálise concluída, mas nenhuma restrição específica foi marcada como violada no loop de diagnóstico. Verifique a lógica.")
    print("-" * 50)


if __name__ == "__main__":
    test_feasibility_check()