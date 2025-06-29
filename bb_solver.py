# bb_solver.py
import gurobipy as gp
from gurobipy import GRB
import time
from typing import Optional, Dict, List
from collections import defaultdict

# Adicionado para usar o objeto Constraint nos cortes
from mip_problem import MIPProblem, Constraint 
from tree_node import Node

class BBSolver:
    def __init__(self, problem: MIPProblem, warmup_nodes: int = 500, log_frequency: int = 100):
        self.problem = problem
        self.best_solution = None
        
        self.best_bound = float('inf') if self.problem.sense == "minimize" else -float('inf')
        self.global_bound = -float('inf') if self.problem.sense == "minimize" else float('inf')
        
        # Mapeamento para acesso rápido às propriedades das variáveis
        self.vars_map = {v.name: v for v in self.problem.variables}
        self.integer_variables = [v.name for v in self.problem.variables if v.is_integer]
        self.binary_variables = {v.name for v in self.problem.variables if v.is_integer and v.lb == 0 and v.ub == 1}

        self.tolerance = 1e-6
        self.node_count = 0
        self.warmup_nodes = warmup_nodes
        self.log_frequency = log_frequency
        self.start_time = 0
        self.pseudocosts = defaultdict(lambda: {'down': {'sum_degrad': 0.0, 'count': 0}, 'up': {'sum_degrad': 0.0, 'count': 0}})
        
        # --- NOVO: Pool de Cortes ---
        # Armazena todos os cortes gerados para reutilizá-los em nós futuros.
        self.cut_pool: List[Constraint] = []

    def _print_header(self):
        print("="*90)
        print(f"Iniciando B&B para o problema: '{self.problem.name}' (Sentido: {self.problem.sense.upper()})")
        print(f"Estratégia de Cortes: Knapsack Cover Cuts")
        print(f"{'Node':>5s}{'Depth':>7s}{'Open':>6s}{'Cuts':>6s}{'NodeBound':>12s}{'BestInt':>12s}{'Gap (%)':>10s}{'Comment':>20s}")
        print("-"*90)

    def _calculate_gap(self) -> float:
        # (Lógica existente - sem alterações)
        is_min = self.problem.sense == "minimize"
        best_int = self.best_bound
        best_node_bound = self.global_bound
        if best_int == float('inf') or best_int == -float('inf'): return float('inf')
        if is_min:
            if abs(best_int) < self.tolerance: return float('inf')
            if best_node_bound >= best_int: return 0.0
            return 100 * (best_int - best_node_bound) / abs(best_int)
        else:
            if abs(best_int) < self.tolerance: return float('inf')
            if best_node_bound <= best_int: return 0.0
            return 100 * (best_node_bound - best_int) / abs(best_int)

    def solve(self):
        self.start_time = time.time()
        self._print_header()
        
        is_min = self.problem.sense == "minimize"
        root_node = Node(depth=0)
        
        # O nó raiz também pode se beneficiar de cortes
        node_stack: List[Node] = [root_node]
        
        while node_stack:

            if node_stack:
                parent_bounds = [n.parent_objective for n in node_stack if n.parent_objective is not None]
                if parent_bounds:
                    if is_min:
                        # Para minimização, o LB global é o mínimo dos bounds dos nós abertos
                        self.global_bound = min(parent_bounds)
                    else:
                        # Para maximização, o UB global é o máximo dos bounds dos nós abertos
                        self.global_bound = max(parent_bounds)
            else:
                # Se a pilha estiver vazia, o processo terminou. O gap é zero.
                self.global_bound = self.best_bound
                
            # --- Lógica de atualização do Global Bound ---
            # (Lógica existente - sem alterações)

            current_node = node_stack.pop()
            self.node_count += 1
            
            # Constrói o modelo Gurobi para o nó atual
            model = self._build_gurobi_model(current_node)
            model.optimize()
            
            comment = ""
            
            # --- NOVO: Loop de Geração de Cortes ---
            if model.Status == GRB.OPTIMAL:
                max_cut_rounds = 10  # Limite para evitar loops excessivos
                for i in range(max_cut_rounds):
                    solution = {v.VarName: v.X for v in model.getVars()}
                    
                    # Verifica se a solução atual é fracionária
                    if self._select_branching_variable(solution) is None:
                        break # Solução já é inteira, não precisa de cortes

                    # Gera novos cortes de cobertura
                    newly_generated_cuts = self._generate_knapsack_cuts(solution)
                    
                    if not newly_generated_cuts:
                        break # Nenhum corte novo foi encontrado

                    print(f"  -> Nó {self.node_count}: Adicionando {len(newly_generated_cuts)} novos cortes de cobertura (rodada {i+1}).")
                    
                    # Adiciona os novos cortes ao modelo e ao pool
                    for cut in newly_generated_cuts:
                        self.cut_pool.append(cut)
                        expr = gp.LinExpr([(c, model.getVarByName(v)) for v, c in cut.coeffs.items()])
                        model.addConstr(expr <= cut.rhs)
                    
                    # Re-otimiza o modelo com os cortes adicionados
                    model.optimize()

                    if model.Status != GRB.OPTIMAL:
                        break # O problema pode se tornar inviável com os cortes
            # --- FIM DO Loop de Geração de Cortes ---
            
            # --- Continuação da Lógica do Nó ---
            if model.Status == GRB.OPTIMAL:
                 self._update_pseudocosts(current_node, model.ObjVal)

            if model.Status != GRB.OPTIMAL:
                comment = "Prune(Infeas/Cut)"
            else:
                if is_min and model.ObjVal >= self.best_bound:
                    comment = "Prune(Bound)"
                elif not is_min and model.ObjVal <= self.best_bound:
                    comment = "Prune(Bound)"

            if self.node_count % self.log_frequency == 0 or comment:
                gap_str = f"{self._calculate_gap():.2f}" if self.best_bound != float('inf') and self.best_bound != -float('inf') else "inf"
                obj_str = f"{model.ObjVal:.2f}" if model.Status == GRB.OPTIMAL else "N/A"
                best_int_str = f"{self.best_bound:.2f}" if self.best_bound != float('inf') and self.best_bound != -float('inf') else "N/A"
                print(f"{self.node_count:>5d}{current_node.depth:>7d}{len(node_stack):>6d}{len(self.cut_pool):>6d}{obj_str:>12s}{best_int_str:>12s}{gap_str:>10s}{comment:>20s}")

            if comment: continue
            
            lp_solution_value = model.ObjVal
            solution = {v.VarName: v.X for v in model.getVars()}
            fractional_var_name = self._select_branching_variable(solution)

            if fractional_var_name is None:
                is_new_best = (is_min and lp_solution_value < self.best_bound) or \
                              (not is_min and lp_solution_value > self.best_bound)
                if is_new_best:
                    self.best_bound = lp_solution_value
                    self.best_solution = solution
                    gap_str = f"{self._calculate_gap():.2f}"
                    print(f"{'':>5s}{'':>7s}{'':>6s}{'':>6s}{'':>12s}{f'>> {self.best_bound:.2f} <<':>12s}{gap_str:>10s}{'>>> New Best <<<':>20s}")
                continue
            else:
                val_to_branch = solution[fractional_var_name]
                new_depth = current_node.depth + 1
                node1 = Node(list(current_node.extra_bounds) + [(fractional_var_name, '<=', float(int(val_to_branch)))], lp_solution_value, fractional_var_name, 'down', new_depth)
                node2 = Node(list(current_node.extra_bounds) + [(fractional_var_name, '>=', float(int(val_to_branch)) + 1)], lp_solution_value, fractional_var_name, 'up', new_depth)
                # Adiciona os nós na pilha
                node_stack.append(node2)
                node_stack.append(node1)

        final_status = "Optimal" if self.best_solution else "Infeasible"
        self._print_summary(final_status)

    def _generate_knapsack_cuts(self, solution: Dict[str, float]) -> List[Constraint]:
        """
        Gera cortes de cobertura (Knapsack Cover Cuts) para restrições do tipo mochila
        que são violados pela solução LP fracionária atual.
        """
        new_cuts = []
        for const in self.problem.constraints:
            # 1. Identificar restrição candidata: sentido '<=', coeficientes > 0, e todas as vars binárias
            if const.sense != '<=': continue
            if not all(v in self.binary_variables for v in const.coeffs.keys()): continue
            if not all(c > 0 for c in const.coeffs.values()): continue

            # 2. Encontrar uma cobertura inicial (greedy)
            # Itens com valor fracionário na solução LP são bons candidatos para a cobertura
            items_in_cover = {v: c for v, c in const.coeffs.items() if solution[v] > self.tolerance}
            
            cover_weight = sum(items_in_cover.values())
            if cover_weight <= const.rhs: continue # Não forma uma cobertura

            # 3. Tornar a cobertura mínima removendo itens redundantes
            minimal_cover = dict(items_in_cover)
            for var_name, coeff in items_in_cover.items():
                if cover_weight - coeff > const.rhs:
                    # Este item é redundante para a cobertura, pode ser removido
                    del minimal_cover[var_name]
                    cover_weight -= coeff
            
            if not minimal_cover: continue

            # 4. Construir o corte e verificar se ele é violado pela solução atual
            # Corte: sum(x_j for j in C) <= |C| - 1
            cut_rhs = len(minimal_cover) - 1
            cut_lhs_val = sum(solution[v] for v in minimal_cover.keys())
            
            if cut_lhs_val > cut_rhs + self.tolerance:
                # O corte é violado, então é útil!
                cut_coeffs = {v: 1.0 for v in minimal_cover.keys()}
                new_cut = Constraint(coeffs=cut_coeffs, sense='<=', rhs=float(cut_rhs))
                
                # Garante que o mesmo corte não seja adicionado múltiplas vezes
                if new_cut not in self.cut_pool and new_cut not in new_cuts:
                    new_cuts.append(new_cut)
                    
        return new_cuts

    def _build_gurobi_model(self, node: Optional[Node] = None) -> gp.Model:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.Model(self.problem.name, env=env)
        sense = GRB.MINIMIZE if self.problem.sense == "minimize" else GRB.MAXIMIZE
        
        gurobi_vars = {v.name: model.addVar(name=v.name, vtype=GRB.CONTINUOUS, lb=v.lb, ub=v.ub) 
                       for v in self.problem.variables}
        
        # Adiciona os bounds específicos do nó de B&B
        if node and node.extra_bounds:
            for var_name, sense_b, value in node.extra_bounds:
                if sense_b == '<=': gurobi_vars[var_name].ub = value
                elif sense_b == '>=': gurobi_vars[var_name].lb = value
        
        # Adiciona a função objetivo
        obj = gp.LinExpr([(c, gurobi_vars[v]) for v, c in self.problem.objective.items()])
        model.setObjective(obj, sense)
        
        # Adiciona as restrições originais
        for const in self.problem.constraints:
            expr = gp.LinExpr([(c, gurobi_vars[v]) for v, c in const.coeffs.items()])
            if const.sense == "<=": model.addConstr(expr <= const.rhs)
            elif const.sense == ">=": model.addConstr(expr >= const.rhs)
            else: model.addConstr(expr == const.rhs)

        # --- NOVO: Adiciona os cortes do pool ao modelo ---
        for cut in self.cut_pool:
            expr = gp.LinExpr([(c, gurobi_vars[v]) for v, c in cut.coeffs.items()])
            model.addConstr(expr <= cut.rhs)
            
        return model

    # --- Métodos restantes (_print_summary, _update_pseudocosts, _select_...) ---
    # Nenhuma alteração necessária nestes métodos.
    # (O código deles é omitido aqui por brevidade, mas deve ser mantido no seu arquivo)

    def _print_summary(self, status: str):
        elapsed_time = time.time() - self.start_time
        print("="*90)
        print("Fim do Processo de Branch & Bound")
        print(f"Status final: {status}")
        print(f"  - Nós explorados : {self.node_count}")
        print(f"  - Cortes gerados : {len(self.cut_pool)}")
        print(f"  - Tempo total    : {elapsed_time:.4f} segundos")
        if self.best_solution:
            print(f"\nMelhor solução inteira encontrada:")
            print(f"  - Valor Objetivo: {self.best_bound:.4f}")
        print("="*90)

    def _update_pseudocosts(self, node: Node, child_lp_value: float):
        if node.parent_objective is None or node.branch_variable is None: return
        degradation = abs(node.parent_objective - child_lp_value)
        if degradation < 0: degradation = 0
        stats = self.pseudocosts[node.branch_variable][node.branch_direction]
        stats['sum_degrad'] += degradation
        stats['count'] += 1

    def _select_most_fractional(self, solution: Dict[str, float]) -> Optional[str]:
        best_var, max_frac = None, -1
        for var_name in self.integer_variables:
            val = solution[var_name]
            if abs(val - round(val)) > self.tolerance:
                frac_dist_from_half = abs(abs(val - int(val)) - 0.5)
                if best_var is None or frac_dist_from_half < max_frac:
                    max_frac = frac_dist_from_half
                    best_var = var_name
        return best_var

    def _select_by_pseudocost(self, solution: Dict[str, float]) -> Optional[str]:
        best_var, best_score = None, -1.0
        fractional_vars = [v for v in self.integer_variables if abs(solution[v] - round(solution[v])) > self.tolerance]
        
        for var_name in fractional_vars:
            down_stats = self.pseudocosts[var_name]['down']
            up_stats = self.pseudocosts[var_name]['up']
            
            down_avg = (down_stats['sum_degrad'] / down_stats['count']) if down_stats['count'] > 0 else 1.0
            up_avg = (up_stats['sum_degrad'] / up_stats['count']) if up_stats['count'] > 0 else 1.0
            
            score = down_avg * up_avg
            if score > best_score:
                best_score = score
                best_var = var_name
        
        return best_var if best_var is not None else self._select_most_fractional(solution)

    def _select_branching_variable(self, solution: Dict[str, float]) -> Optional[str]:
        if self.node_count <= self.warmup_nodes:
            return self._select_most_fractional(solution)
        else:
            return self._select_by_pseudocost(solution)