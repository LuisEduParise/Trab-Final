# problem_parser.py
import re
from typing import List, Dict
import gurobipy as gp
from gurobipy import GRB

from mip_problem import MIPProblem, Variable, Constraint

def _parse_linear_expression(expr_str: str, variables_map: Dict[str, Variable]) -> Dict[str, float]:
    """Analisa uma expressão linear (ex: '50x_A + 20x_B') e retorna um dicionário de coeficientes."""
    coeffs = {}
    expr_str = expr_str.replace(' - ', ' + -')
    terms = expr_str.split('+')

    for term in terms:
        term = term.strip()
        if not term:
            continue

        match = re.match(r'([-+]?\s*\d*\.?\d*)\s*\*?\s*(x_[a-zA-Z0-9_]+)', term)
        if match:
            coeff_str, var_name = match.groups()
            coeff_str = coeff_str.strip().replace(' ', '')
            
            if coeff_str == '' or coeff_str == '+':
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)
            
            coeffs[var_name] = coeff
            
            if var_name not in variables_map:
                variables_map[var_name] = Variable(name=var_name)
    return coeffs

def create_problem_from_file(file_path: str) -> MIPProblem:
    """
    Lê um arquivo de definição de problema e o converte em um objeto MIPProblem.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Lê todas as linhas, removendo espaços em branco extras e linhas vazias
        lines = [line.strip() for line in f if line.strip()]

    problem_name = "Unnamed Problem"
    objective_sense = "minimize"
    objective_coeffs = {}
    constraints: List[Constraint] = []
    variables_map: Dict[str, Variable] = {}
    
    current_section = None
    
    # --- INÍCIO DA CORREÇÃO DEFINITIVA ---
    # Variável para acumular as partes de uma restrição que abrange várias linhas
    active_constraint_str = ""

    for line in lines:
        is_new_section = line.startswith('[') and line.endswith(']')

        # Se estamos na seção de restrições e encontramos uma nova seção ou uma nova restrição (com ':'),
        # então a restrição anterior terminou e deve ser processada.
        if current_section == "CONSTRAINTS" and (is_new_section or ':' in line):
            if active_constraint_str:
                const_name, const_body = active_constraint_str.split(':', 1)
                match = re.search(r'(<=|>=|==)', const_body)
                if match:
                    sense = match.group(1)
                    expr_part, rhs_part = re.split(r'\s*(<=|>=|==)\s*', const_body, 1)[::2]
                    const_coeffs = _parse_linear_expression(expr_part, variables_map)
                    rhs = float(rhs_part)
                    constraints.append(Constraint(coeffs=const_coeffs, sense=sense, rhs=rhs))
                active_constraint_str = "" # Limpa para a próxima

        # Lógica para trocar de seção
        if is_new_section:
            current_section = line[1:-1].upper()
            continue

        # Lógica de processamento para cada seção
        if current_section == "NAME":
            problem_name = line

        elif current_section == "OBJECTIVE":
            sense_part, expr_part = line.split(':', 1)
            objective_sense = "maximize" if "max" in sense_part else "minimize"
            objective_coeffs = _parse_linear_expression(expr_part, variables_map)

        elif current_section == "CONSTRAINTS":
            # Se a linha tem ':', ela inicia (ou substitui) a restrição ativa.
            # Se não tem ':', ela é uma continuação e é anexada.
            if ':' in line:
                active_constraint_str = line
            else:
                active_constraint_str += " " + line

        elif current_section == "BINARY":
            var_name = line.strip()
            if var_name in variables_map:
                variables_map[var_name].is_integer = True
                variables_map[var_name].lb = 0.0
                variables_map[var_name].ub = 1.0
            else:
                variables_map[var_name] = Variable(name=var_name, is_integer=True, lb=0.0, ub=1.0)

    # Processa a última restrição que pode ter ficado pendente após o fim do laço
    if active_constraint_str:
        const_name, const_body = active_constraint_str.split(':', 1)
        match = re.search(r'(<=|>=|==)', const_body)
        if match:
            sense = match.group(1)
            expr_part, rhs_part = re.split(r'\s*(<=|>=|==)\s*', const_body, 1)[::2]
            const_coeffs = _parse_linear_expression(expr_part, variables_map)
            rhs = float(rhs_part)
            constraints.append(Constraint(coeffs=const_coeffs, sense=sense, rhs=rhs))
    # --- FIM DA CORREÇÃO DEFINITIVA ---

    final_variables = list(variables_map.values())

    return MIPProblem(
        name=problem_name,
        variables=final_variables,
        objective=objective_coeffs,
        constraints=constraints,
        sense=objective_sense
    )

def create_problem_from_mps(filepath: str) -> MIPProblem:
    """
    Lê um arquivo no formato .MPS e o converte para um objeto MIPProblem.

    Args:
        filepath: O caminho para o arquivo .MPS.

    Returns:
        Um objeto MIPProblem representando o problema lido.
    """
    print(f"Lendo o arquivo MPS: {filepath}...")

    # 1. Use Gurobi para ler o arquivo .MPS
    # Gurobi lida com a complexidade de analisar o formato do arquivo.
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0) # Suprime a saída do Gurobi
    env.start()
    
    gurobi_model = gp.read(filepath, env=env)
    gurobi_model.update()

    # 2. Extrair informações do modelo Gurobi e traduzir para nossas estruturas
    
    # Nome do problema
    problem_name = gurobi_model.ModelName

    # Sentido da otimização
    sense = "minimize" if gurobi_model.ModelSense == GRB.MINIMIZE else "maximize"

    # Extrair Variáveis
    problem_variables = []
    for v in gurobi_model.getVars():
        var = Variable(
            name=v.VarName,
            is_integer=(v.VType == GRB.INTEGER or v.VType == GRB.BINARY),
            lb=v.LB,
            ub=v.UB
        )
        problem_variables.append(var)

    # Extrair Função Objetivo
    objective_coeffs = {}
    obj_expr = gurobi_model.getObjective()
    if isinstance(obj_expr, gp.LinExpr):
        for i in range(obj_expr.size()):
            var = obj_expr.getVar(i)
            coeff = obj_expr.getCoeff(i)
            objective_coeffs[var.VarName] = coeff

    # Extrair Restrições
    problem_constraints = []
    for c in gurobi_model.getConstrs():
        row = gurobi_model.getRow(c)
        
        coeffs = {}
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            coeffs[var.VarName] = coeff
            
        sense_map = {
            GRB.LESS_EQUAL: '<=',
            GRB.GREATER_EQUAL: '>=',
            GRB.EQUAL: '=='
        }
        
        constraint = Constraint(
            coeffs=coeffs,
            sense=sense_map.get(c.Sense, '=='), # Mapeia o sentido do Gurobi para o nosso
            rhs=c.RHS
        )
        problem_constraints.append(constraint)

    print("Arquivo MPS lido e convertido com sucesso.")
    
    # 3. Criar e retornar o objeto MIPProblem final
    return MIPProblem(
        name=problem_name,
        variables=problem_variables,
        objective=objective_coeffs,
        constraints=problem_constraints,
        sense=sense
    )