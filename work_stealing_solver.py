# work_stealing_solver.py
"""
Implementa o resolvedor principal de MIP usando uma arquitetura paralela de
Branch and Cut com balanceamento de carga por Work Stealing.

Este módulo contém as classes que definem a lógica de um worker individual,
o processamento de um nó da árvore de busca e a orquestração geral do
processo de resolução.
"""
import multiprocessing as mp
import time
import random
import heapq
from queue import Empty
from typing import List, Dict, Optional, Tuple
import traceback
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

from mip_problem import MIPProblem, Constraint
from tree_node import Node
from parallel_utils import SharedState
from multiprocessing.synchronize import Event as MpEvent


class NodeProcessor:
    """
    Encapsula toda a lógica para processar um único nó da árvore de B&B.

    Isso inclui resolver a relaxação linear (LP), executar heurísticas primais,
    gerar cortes (cutting planes) e selecionar variáveis para o branching.
    """
    def __init__(self, problem: MIPProblem, verbose: bool = True):
        """
        Inicializa o processador de nós.

        Args:
            problem (MIPProblem): A definição do problema de MIP a ser resolvido.
            verbose (bool): Controla se mensagens de log devem ser impressas.
        """
        self.problem = problem
        self.verbose = verbose  # Armazena o parâmetro
        self.vars_map = {var.name: var for var in self.problem.variables}
        self.integer_variables = [var.name for var in self.problem.variables if var.is_integer]
        self.binary_variables = {var.name for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1}
        self.tolerance = 1e-6
        self.pseudocosts = defaultdict(lambda: {'down': {'sum_degrad': 0.0, 'count': 0}, 'up': {'sum_degrad': 0.0, 'count': 0}})
        self.warmup_nodes = 500
        self.local_node_count = 0
        self._initialize_pseudocosts()

    def _initialize_pseudocosts(self):
        """
        Inicializa os pseudocustos com base nos coeficientes da função objetivo.
        """
        for var_name in self.integer_variables:
            obj_coeff = abs(self.problem.objective.get(var_name, 0.0))
            initial_value = 1e-6 + obj_coeff
            self.pseudocosts[var_name]['down'] = {'sum_degrad': initial_value, 'count': 1}
            self.pseudocosts[var_name]['up'] = {'sum_degrad': initial_value, 'count': 1}

    def run_feasibility_pump(self) -> Optional[Dict]:
        """
        Executa a heurística Feasibility Pump com limites para evitar bloqueios.
        """
        print("[Feasibility Pump]: Iniciando busca por solução inicial...")
        root_node = Node()
        model = self._build_gurobi_model(root_node, [])
        model.setParam(GRB.Param.TimeLimit, 15.0)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            print("[Feasibility Pump]: LP inicial inviável."); return None
        
        for i in range(20):
            lp_solution = {v.VarName: v.X for v in model.getVars()}
            if self._select_most_fractional(lp_solution) is None:
                if self._is_solution_feasible(lp_solution):
                    cost = self._calculate_solution_cost(lp_solution)
                    print(f"[Feasibility Pump]: Solução ótima do LP é inteira! Custo: {cost:.2f}")
                    return {'type': 'solution', 'value': cost, 'solution': lp_solution}
                break
            
            rounded_solution = {name: round(val) for name, val in lp_solution.items() if name in self.integer_variables}
            rounded_solution.update({name: val for name, val in lp_solution.items() if name not in self.integer_variables})
            
            if self._is_solution_feasible(rounded_solution):
                cost = self._calculate_solution_cost(rounded_solution)
                print(f"[Feasibility Pump]: Solução viável encontrada com custo {cost:.2f}!")
                return {'type': 'solution', 'value': cost, 'solution': rounded_solution}
            
            dist_model = self._build_distance_model(rounded_solution)
            dist_model.setParam(GRB.Param.TimeLimit, 15.0)
            dist_model.optimize()
            
            if dist_model.Status != GRB.OPTIMAL: break
            model = dist_model
            
        print("[Feasibility Pump]: Heurística encerrada sem solução.")
        return None

    def run_rins_heuristic(self, lp_solution: Dict[str, float], incumbent_solution: Dict[str, float], shared_state: 'SharedState', num_main_workers: int):
        """
        Executa a heurística RINS (Relaxation Induced Neighborhood Search).
        """
        if not incumbent_solution:
            return

        #if self.verbose:
            #print("[RINS]: Iniciando heurística...")
        
        sub_problem = self.problem.copy()
        sub_vars_map = {var.name: var for var in sub_problem.variables}
        
        num_fixed = 0
        num_free = 0

        for var_name in self.integer_variables:
            lp_val = lp_solution.get(var_name)
            inc_val = incumbent_solution.get(var_name)

            if lp_val is not None and inc_val is not None:
                lp_val_rounded = round(lp_val)
                
                if abs(lp_val_rounded - inc_val) < self.tolerance:
                    var_in_subproblem = sub_vars_map[var_name]
                    var_in_subproblem.lb = lp_val_rounded
                    var_in_subproblem.ub = lp_val_rounded
                    num_fixed += 1
                else:
                    num_free += 1
        
        if num_free == 0 or num_free > 0.5 * len(self.integer_variables):
            #if self.verbose:
                #print(f"[RINS]: Pulo. (Livres: {num_free}, Fixas: {num_fixed})")
            return

        #if self.verbose:
            #print(f"[RINS]: Resolvendo sub-MIP com {num_free} variáveis livres e {num_fixed} fixas...")

        sub_solver = SerialBranchAndBoundSolver(
            problem=sub_problem,
            node_limit=1000,
            timeout=15.0
        )
        
        new_cost, new_solution = sub_solver.solve()

        if new_solution:
            if shared_state.update_best_solution(new_cost, new_solution):
                pass
                #if self.verbose:
                    #print(f"[RINS]: SUCESSO! Nova solução encontrada com custo {new_cost:.4f}")
            pass
            #else:
                #if self.verbose:
                    #print("[RINS]: Sub-MIP resolvido, mas sem melhoria na solução.")

    def _is_solution_feasible(self, solution: Dict[str, float]) -> bool:
        for var_name, var_value in solution.items():
            var_def = self.vars_map.get(var_name)
            if var_def:
                if var_value < var_def.lb - self.tolerance or var_value > var_def.ub + self.tolerance:
                    return False
        for const in self.problem.constraints:
            activity = sum(coeff * solution.get(var, 0) for var, coeff in const.coeffs.items())
            if const.sense == '<=' and activity > const.rhs + self.tolerance: return False
            if const.sense == '>=' and activity < const.rhs - self.tolerance: return False
            if const.sense == '==' and abs(activity - const.rhs) > self.tolerance: return False
        return True
    
    def _calculate_solution_cost(self, solution: Dict[str, float]) -> float:
        return sum(coeff * solution.get(var, 0) for var, coeff in self.problem.objective.items())

    def _build_distance_model(self, rounded_solution: Dict[str, float]) -> gp.Model:
        env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
        model = gp.Model("distance_lp", env=env)
        gurobi_vars = {v.name: model.addVar(name=v.name, vtype=GRB.CONTINUOUS, lb=v.lb, ub=v.ub) for v in self.problem.variables}
        dist_vars, objective = {}, gp.LinExpr()
        for var_name in self.integer_variables:
            dist_vars[var_name] = model.addVar(name=f"dist_{var_name}")
            objective += dist_vars[var_name]
            rounded_val = rounded_solution[var_name]
            model.addConstr(dist_vars[var_name] >= gurobi_vars[var_name] - rounded_val)
            model.addConstr(dist_vars[var_name] >= rounded_val - gurobi_vars[var_name])
        model.setObjective(objective, GRB.MINIMIZE)
        for const in self.problem.constraints:
            expr = gp.LinExpr([(c, gurobi_vars[v]) for v, c in const.coeffs.items()])
            if const.sense == "<=": model.addConstr(expr <= const.rhs)
            elif const.sense == ">=": model.addConstr(expr >= const.rhs)
            else: model.addConstr(expr == const.rhs)
        return model

    def _update_pseudocosts(self, node: Node, child_lp_value: float):
        if node.lp_bound is None or node.branch_variable is None: return
        degradation = abs(node.lp_bound - child_lp_value)
        if degradation < self.tolerance: degradation = 0.0
        
        stats = self.pseudocosts[node.branch_variable][node.branch_direction]
        stats['sum_degrad'] += degradation
        stats['count'] += 1

    def _select_by_pseudocost(self, solution: Dict[str, float]) -> Optional[str]:
        RELIABILITY_THRESHOLD = 5
        best_var, best_score = None, -1.0
        fractional_vars = [v for v in self.integer_variables if abs(solution[v] - round(solution[v])) > self.tolerance]
        if not fractional_vars: return None
        for var_name in fractional_vars:
            val = solution[var_name]; f_down = val - int(val); f_up = 1.0 - f_down
            down_stats, up_stats = self.pseudocosts[var_name]['down'], self.pseudocosts[var_name]['up']
            pc_down = (down_stats['sum_degrad'] / down_stats['count']) if down_stats['count'] > 0 else 1.0
            pc_up = (up_stats['sum_degrad'] / up_stats['count']) if up_stats['count'] > 0 else 1.0
            
            total_count = down_stats['count'] + up_stats['count']
            reliability = min(1.0, total_count / (2 * RELIABILITY_THRESHOLD))
            score = reliability * ((f_down * pc_down) + (f_up * pc_up))

            if score > best_score:
                best_score, best_var = score, var_name
        return best_var
    
    def _run_coefficient_diving(self, initial_model: gp.Model, max_dive_depth: int = 20) -> Optional[Dict]:
        dive_model = initial_model.copy()
        for depth in range(max_dive_depth):
            dive_model.optimize()
            if dive_model.Status != GRB.OPTIMAL: return None
            solution = {v.VarName: v.X for v in dive_model.getVars()}
            frac_vars = {v: solution[v] for v in self.integer_variables if abs(solution[v] - round(solution[v])) > self.tolerance}
            if not frac_vars:
                if self._is_solution_feasible(solution):
                    cost = self._calculate_solution_cost(solution)
                    print(f"[CoefficientDiving]: Solução viável encontrada com custo {cost:.2f}!")
                    return {'type': 'solution', 'value': cost, 'solution': solution}
                return None
            best_var_name, min_locks = None, float('inf')
            for var_name, frac_value in frac_vars.items():
                var_def = self.vars_map[var_name]
                if frac_value - int(frac_value) < 0.5:
                    if var_def.down_locks < min_locks: min_locks, best_var_name = var_def.down_locks, var_name
                else:
                    if var_def.up_locks < min_locks: min_locks, best_var_name = var_def.up_locks, var_name
            if best_var_name is None: return None
            var_to_fix = dive_model.getVarByName(best_var_name)
            rounded_val = round(var_to_fix.X)
            var_to_fix.lb, var_to_fix.ub = rounded_val, rounded_val
        return None
    
    def _run_rounding_heuristic(self, model: gp.Model) -> Optional[Dict]:
        try:
            lp_solution = {v.VarName: v.X for v in model.getVars()}
            rounded_solution = {name: round(val) if name in self.integer_variables else val for name, val in lp_solution.items()}
            if self._is_solution_feasible(rounded_solution):
                cost = self._calculate_solution_cost(rounded_solution)
                return {'type': 'solution', 'value': cost, 'solution': rounded_solution}
        except Exception:
            pass
        return None

    def _select_most_fractional(self, solution: Dict[str, float]) -> Optional[str]:
        best_var, max_frac_dist = None, -1.0
        for var_name in self.integer_variables:
            val = solution[var_name]
            if abs(val - round(val)) > self.tolerance:
                frac_dist_from_half = abs(abs(val - int(val)) - 0.5)
                if best_var is None or frac_dist_from_half < max_frac_dist:
                    max_frac_dist, best_var = frac_dist_from_half, var_name
        return best_var
    
    def _run_diving_heuristic(self, initial_model: gp.Model, max_dive_depth: int = 10) -> Optional[Dict]:
        dive_model = initial_model.copy()
        for depth in range(max_dive_depth):
            dive_model.optimize()
            if dive_model.Status != GRB.OPTIMAL: return None
            solution = {v.VarName: v.X for v in dive_model.getVars()}
            frac_vars = [v for v in self.integer_variables if abs(solution[v] - round(solution[v])) > self.tolerance]
            if not frac_vars:
                cost = self._calculate_solution_cost(solution)
                print(f"[Diving Heuristic]: Solução viável encontrada com custo {cost:.2f}!")
                return {'type': 'solution', 'value': cost, 'solution': solution}
            best_var_name = max(frac_vars, key=lambda v_name: abs(solution[v_name] - round(solution[v_name])))
            var_to_fix = dive_model.getVarByName(best_var_name)
            rounded_val = round(var_to_fix.X)
            var_to_fix.lb, var_to_fix.ub = rounded_val, rounded_val
        return None

    def _select_branching_variable(self, solution: Dict[str, float]) -> Optional[str]:
        if self.local_node_count <= self.warmup_nodes:
            return self._select_most_fractional(solution)
        else:
            return self._select_by_pseudocost(solution) or self._select_most_fractional(solution)

    def process_node(self, node: Node, best_bound_so_far: float, cut_pool: List[Constraint]) -> List[Dict]:
        self.local_node_count += 1
        model = self._build_gurobi_model(node, cut_pool)
        model.optimize()
        
        results, is_min = [], self.problem.sense == "minimize"
        
        if self.local_node_count % 50 == 0 and model.Status == GRB.OPTIMAL:
            rounding_result = self._run_rounding_heuristic(model)
            if rounding_result:
                results.append(rounding_result)

        if model.Status == GRB.OPTIMAL:
            self._update_pseudocosts(node, model.ObjVal)
            for _ in range(5):
                solution = {v.VarName: v.X for v in model.getVars()}
                if self._select_most_fractional(solution) is None: break
                new_cuts = self._generate_knapsack_cuts(solution, cut_pool)
                if not new_cuts: break
                results.extend([{'type': 'cut', 'cut': c} for c in new_cuts])
                for cut in new_cuts:
                    expr = gp.LinExpr([(c, model.getVarByName(v)) for v, c in cut.coeffs.items()])
                    model.addConstr(expr <= cut.rhs)
                model.optimize()

        if model.Status != GRB.OPTIMAL:
            return results

        lp_solution_value = model.ObjVal
        
        if (is_min and lp_solution_value >= best_bound_so_far) or \
           (not is_min and lp_solution_value <= best_bound_so_far):
            return results

        solution = {v.VarName: v.X for v in model.getVars()}
        fractional_var = self._select_branching_variable(solution)

        if fractional_var is None:
            results.append({'type': 'solution', 'value': lp_solution_value, 'solution': solution})
        else:
            val_to_branch = solution[fractional_var]
            new_depth = node.depth + 1
            objective_for_heap = lp_solution_value if is_min else -lp_solution_value
            
            node1 = Node(lp_bound=lp_solution_value, parent_objective=objective_for_heap,
                         extra_bounds=list(node.extra_bounds) + [(fractional_var, '<=', float(int(val_to_branch)))], 
                         branch_variable=fractional_var, branch_direction='down', depth=new_depth)
            node2 = Node(lp_bound=lp_solution_value, parent_objective=objective_for_heap,
                         extra_bounds=list(node.extra_bounds) + [(fractional_var, '>=', float(int(val_to_branch)) + 1)], 
                         branch_variable=fractional_var, branch_direction='up', depth=new_depth)
            
            results.append({'type': 'node', 'node': node1})
            results.append({'type': 'node', 'node': node2})
        return results
            
    def _build_gurobi_model(self, node: Node, cut_pool: List[Constraint]) -> gp.Model:
        env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
        model = gp.Model(self.problem.name, env=env)
        sense = GRB.MINIMIZE if self.problem.sense == "minimize" else GRB.MAXIMIZE
        
        gurobi_vars = {v.name: model.addVar(name=v.name, vtype=GRB.CONTINUOUS, lb=v.lb, ub=v.ub) for v in self.problem.variables}
        
        if node and node.extra_bounds:
            for var_name, sense_b, value in node.extra_bounds:
                if sense_b == '<=': gurobi_vars[var_name].ub = value
                elif sense_b == '>=': gurobi_vars[var_name].lb = value
                
        obj = gp.LinExpr([(c, gurobi_vars[v]) for v, c in self.problem.objective.items()])
        model.setObjective(obj, sense)
        
        for const in self.problem.constraints:
            expr = gp.LinExpr([(c, gurobi_vars[v]) for v, c in const.coeffs.items()])
            if const.sense == "<=": model.addConstr(expr <= const.rhs)
            elif const.sense == ">=": model.addConstr(expr >= const.rhs)
            else: model.addConstr(expr == const.rhs)
        
        for cut in cut_pool:
            expr = gp.LinExpr([(c, gurobi_vars[v]) for v, c in cut.coeffs.items()])
            model.addConstr(expr <= cut.rhs)
        return model

    def _generate_knapsack_cuts(self, solution: Dict[str, float], cut_pool: List[Constraint]) -> List[Constraint]:
        """
        Gera cortes de cobertura (knapsack cover cuts) para restrições binárias.

        Esta é uma implementação simplificada que encontra um 'minimal cover'
        violado e gera o corte correspondente, conforme a Seção 8.1 da tese.
        """
        new_cuts = []
        for const in self.problem.constraints:
            # Passo 1: Identificar restrições knapsack candidatas
            # (Assume que o 'upgrader' do presolve poderia ter adicionado um .special_type)
            is_knapsack_like = (
                const.sense == '<=' and
                all(v in self.binary_variables for v in const.coeffs.keys()) and
                all(c > self.tolerance for c in const.coeffs.values())
            )
            if not is_knapsack_like:
                continue

            # Passo 2: Encontrar um 'cover' a partir da solução LP
            # Itens são as variáveis binárias com valor > 0 na solução LP
            items_in_lp_solution = {v: c for v, c in const.coeffs.items() if solution[v] > self.tolerance}
            
            # Se a soma dos pesos dos itens na solução LP não viola a capacidade, não há cover
            if sum(items_in_lp_solution.values()) <= const.rhs:
                continue
            
            # Ordena os itens por peso para a heurística greedy
            sorted_cover_items = sorted(items_in_lp_solution.items(), key=lambda item: item[1])
            cover_weight = sum(w for _, w in sorted_cover_items)

            # Passo 3: Encontrar uma cobertura minimal (greedy)
            # Remove itens desnecessários do cover, começando pelo de menor peso
            minimal_cover_items = dict(sorted_cover_items)
            for var_name, coeff in sorted_cover_items:
                if cover_weight - coeff > const.rhs:
                    del minimal_cover_items[var_name]
                    cover_weight -= coeff
            
            if not minimal_cover_items:
                continue
            
            # Passo 4: Gerar o corte
            # O corte é: sum(x_j for j in minimal_cover) <= |minimal_cover| - 1
            cut_rhs = len(minimal_cover_items) - 1
            cut_coeffs = {var_name: 1.0 for var_name in minimal_cover_items.keys()}

            # Passo 5: Verificar se o corte é violado
            cut_activity = sum(solution[v] for v in cut_coeffs.keys())
            if cut_activity > cut_rhs + self.tolerance:
                new_cut = Constraint(coeffs=cut_coeffs, sense='<=', rhs=float(cut_rhs))
                
                # Adiciona apenas se o corte for novo
                # (A verificação de duplicatas no cut_pool global é feita no SharedState)
                new_cuts.append(new_cut)

        return new_cuts


# Em work_stealing_solver.py

class ParallelWorker:
    """
    Representa um worker que processa nós da árvore de B&B em paralelo.
    """
    def __init__(self, worker_id: int, problem: MIPProblem, shared_state: SharedState, 
                 work_queues: List[mp.Queue], termination_event: MpEvent, 
                 switch_event: MpEvent, enable_heuristics: bool):
        """
        Inicializa o worker.
        """
        self.worker_id = worker_id
        self.problem = problem
        self.shared_state = shared_state
        self.work_queues = work_queues
        self.termination_event = termination_event
        self.switch_to_bb_event = switch_event
        self.my_queue = self.work_queues[self.worker_id]
        # Passa o flag 'verbose' para o NodeProcessor
        self.node_processor = NodeProcessor(self.problem, verbose=enable_heuristics)
        self.local_heap: List[Node] = []
        self.is_idle = False
        self.num_workers = len(work_queues)
        self.enable_heuristics = enable_heuristics
        # A impressão de log do worker também é controlada pelo flag
        if self.enable_heuristics:
            print(f"[Worker {self.worker_id}]: Iniciado.")

    def run(self):
        """
        O loop principal de execução do worker.
        """
        # Define a estratégia de busca inicial
        Node.strategy = "best_bound"
        if self.worker_id == 0 and self.enable_heuristics:
            print(f"[Worker 0]: Todos os workers iniciando com a estratégia Best-Bound.")

        # Worker 0 executa heurísticas iniciais e cria o nó raiz
        if self.worker_id == 0:
            if self.enable_heuristics:
                initial_solution = self.node_processor.run_feasibility_pump()
                if initial_solution: 
                    self.shared_state.update_best_solution(initial_solution['value'], initial_solution['solution'])
            
            is_min = self.problem.sense == 'minimize'
            initial_bound = float('-inf') if is_min else float('inf')
            root_node = Node(lp_bound=initial_bound, parent_objective=float('-inf'), depth=0)
            heapq.heappush(self.local_heap, root_node)
            
        while not self.termination_event.is_set():
            if self.switch_to_bb_event.is_set() and Node.strategy != "best_bound":
                if self.enable_heuristics:
                    print(f"[Worker {self.worker_id}]: Sinal do Monitor recebido. Trocando para estratégia Best-Bound.")
                Node.strategy = "best_bound"
                heapq.heapify(self.local_heap)

            self._handle_incoming_messages()

            if self.local_heap:
                if self.is_idle:
                    self.shared_state.decrement_idle_worker_count(); self.is_idle = False
                
                best_node_in_heap = self.local_heap[0]
                if best_node_in_heap.lp_bound is not None:
                    self.shared_state.update_worker_best_bound(self.worker_id, best_node_in_heap.lp_bound)
                
                node_to_process = heapq.heappop(self.local_heap)
                current_global_cuts = self.shared_state.get_cuts()

                results = self.node_processor.process_node(node_to_process, self.shared_state.get_best_cost(), current_global_cuts)
                self.shared_state.increment_nodes_processed()
                nodes_processed = self.shared_state.nodes_processed.value

                if self.enable_heuristics:
                    model = self.node_processor._build_gurobi_model(node_to_process, current_global_cuts)
                    model.optimize()

                    if not self.shared_state.has_solution.value and model.Status == GRB.OPTIMAL:
                        heuristic_to_run = None
                        if nodes_processed % 1000 == 0: heuristic_to_run = self.node_processor._run_coefficient_diving
                        elif nodes_processed % 500 == 0: heuristic_to_run = self.node_processor._run_diving_heuristic
                        if heuristic_to_run:
                            heuristic_result = heuristic_to_run(model)
                            if heuristic_result:
                                self.shared_state.update_best_solution(heuristic_result['value'], heuristic_result['solution'])

                    if self.shared_state.has_solution.value and nodes_processed % 1000 == 0 and model.Status == GRB.OPTIMAL:
                        lp_solution = {v.VarName: v.X for v in model.getVars()}
                        incumbent_solution = self.shared_state.get_best_solution()
                        self.node_processor.run_rins_heuristic(lp_solution, incumbent_solution, self.shared_state, self.num_workers)

                new_cuts_found = []
                for res in results:
                    if res['type'] == 'node': heapq.heappush(self.local_heap, res['node'])
                    elif res['type'] == 'solution': self.shared_state.update_best_solution(res['value'], res['solution'])
                    elif res['type'] == 'cut': new_cuts_found.append(res['cut'])
                if new_cuts_found: self.shared_state.add_cuts(new_cuts_found)
            else:
                if not self.is_idle:
                    self.shared_state.increment_idle_worker_count(); self.is_idle = True
                    invalid_bound = float('inf') if self.problem.sense == 'minimize' else -float('inf')
                    self.shared_state.update_worker_best_bound(self.worker_id, invalid_bound)
                self._request_work()
                time.sleep(0.1 + random.uniform(0, 0.2))
                
        if self.enable_heuristics:
            print(f"[Worker {self.worker_id}]: Sinal de término recebido. Encerrando.")
    
    def _handle_incoming_messages(self):
        """Processa mensagens recebidas na fila do worker (pedidos de roubo, etc.)."""
        while not self.my_queue.empty():
            try:
                message = self.my_queue.get_nowait()
                msg_type = message.get('type')
                if msg_type == 'STEAL_REQUEST':
                    self._handle_steal_request(message['from_id'])
                elif msg_type == 'WORK_RESPONSE':
                    self._handle_work_response(message['nodes'])
            except Empty:
                return

    def _handle_steal_request(self, requester_id: int):
        """
        Lida com um pedido de trabalho de outro worker.
        Se tiver trabalho suficiente, doa metade dos seus "piores" nós
        (nós com piores bounds) para o worker solicitante.
        """
        if len(self.local_heap) > 1:
            num_to_share = len(self.local_heap) // 2
            
            sorted_nodes = sorted(self.local_heap)
            num_to_keep = len(self.local_heap) - num_to_share
            
            nodes_to_keep = sorted_nodes[:num_to_keep]
            nodes_to_share = sorted_nodes[num_to_keep:]
            
            self.local_heap = nodes_to_keep
            heapq.heapify(self.local_heap)
            
            if nodes_to_share:
                self.work_queues[requester_id].put({'type': 'WORK_RESPONSE', 'nodes': nodes_to_share})

    def _handle_work_response(self, nodes: List[Node]):
        """Lida com a recepção de nós de outro worker após um pedido de roubo."""
        if nodes:
            for node in nodes:
                heapq.heappush(self.local_heap, node)

    def _request_work(self):
        """Envia um pedido de trabalho (steal request) para um worker aleatório."""
        target_id = random.choice([i for i in range(self.num_workers) if i != self.worker_id])
        self.work_queues[target_id].put({'type': 'STEAL_REQUEST', 'from_id': self.worker_id})


def worker_entry_point(worker_id: int, problem: MIPProblem, shared_state: SharedState, 
                    work_queues: List[mp.Queue], termination_event: MpEvent, 
                    switch_event: MpEvent, enable_heuristics: bool):
    try:
        worker = ParallelWorker(worker_id, problem, shared_state, work_queues, 
                                termination_event, switch_event, enable_heuristics)
        worker.run()
    except Exception as e:
        print(f"[Worker {worker_id}]: ERRO FATAL - {e}")
        traceback.print_exc()


class SerialBranchAndBoundSolver:
    """
    Um resolvedor de MIP serial e simplificado, projetado para ser usado
    dentro de heurísticas como a RINS. Ele executa um loop de Branch & Bound
    básico em uma única thread.
    """
    def __init__(self, problem: MIPProblem, node_limit: int = 1000, timeout: float = 15.0):
        self.problem = problem
        self.node_limit = node_limit
        self.timeout = timeout
        self.node_processor = NodeProcessor(self.problem)

    def solve(self) -> Tuple[float, Dict[str, float]]:
        start_time = time.time()
        
        # Estado local do solver serial
        is_min = self.problem.sense == "minimize"
        best_cost = float('inf') if is_min else -float('inf')
        best_solution = {}
        
        # Fila de prioridade de nós (heap)
        open_nodes = []
        initial_bound = float('-inf') if is_min else float('inf')
        root_node = Node(lp_bound=initial_bound, parent_objective=float('-inf'), depth=0)
        heapq.heappush(open_nodes, root_node)
        
        nodes_processed = 0

        while open_nodes and nodes_processed < self.node_limit and (time.time() - start_time) < self.timeout:
            node_to_process = heapq.heappop(open_nodes)
            
            # Processa o nó usando o mesmo NodeProcessor
            results = self.node_processor.process_node(node_to_process, best_cost, [])
            nodes_processed += 1
            
            for res in results:
                if res['type'] == 'node':
                    heapq.heappush(open_nodes, res['node'])
                elif res['type'] == 'solution':
                    cost = res['value']
                    is_better = (is_min and cost < best_cost) or (not is_min and cost > best_cost)
                    if is_better:
                        best_cost = cost
                        best_solution = res['solution']
        
        return best_cost, best_solution


class WorkStealingSolver:
    """
    Orquestra a resolução de um MIP usando a estratégia de work-stealing.
    """
    def __init__(self, problem: MIPProblem, num_workers: Optional[int] = None, 
                 timeout: Optional[float] = None, stagnation_limit: Optional[int] = None, 
                 mip_gap_tolerance: float = 1e-4,
                 node_limit: Optional[int] = None,
                 verbose: bool = True,
                 enable_heuristics: bool = True):
        self.problem = problem
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.timeout = timeout
        self.stagnation_limit = stagnation_limit
        self.mip_gap_tolerance = mip_gap_tolerance
        self.node_limit = node_limit
        self.verbose = verbose
        self.enable_heuristics = enable_heuristics

    def _calculate_gap(self, best_primal: float, best_dual: float) -> float:
        """Calcula o gap de otimalidade percentual."""
        is_min = self.problem.sense == "minimize"
        if (is_min and best_primal == float('inf')) or (not is_min and best_primal == -float('inf')) or \
           (is_min and best_dual == -float('inf')) or (not is_min and best_dual == float('inf')):
            return float('inf')
        if abs(best_primal) < 1e-9:
            return float('inf') if abs(best_primal - best_dual) > 1e-9 else 0.0
        return 100.0 * abs(best_primal - best_dual) / (abs(best_primal) + 1e-9)

    def solve(self) -> Tuple[float, Dict[str, float]]:
        """
        Inicia e gerencia o processo de resolução paralela.
        """
        start_time = time.time()
        termination_reason = "Desconhecido"
        
        if self.verbose:
            print("="*100)
            print(f"Iniciando Solver com Work Stealing para '{self.problem.name}' com {self.num_workers} workers.")
            print(f"{'Nodes':>8s}{'Idle':>6s}{'Cuts':>6s}{'BestInt':>15s}{'BestBound':>15s}{'Gap(%)':>10s}{'Time(s)':>10s}{'Update':>15s}")
            print("-"*100)
        
        workers = []
        try:
            shared_state = SharedState(self.num_workers, self.problem.sense)
            termination_event = mp.Event()
            switch_to_bb_event = mp.Event()
            work_queues = [mp.Queue() for _ in range(self.num_workers)]
            
            for i in range(self.num_workers):
                p = mp.Process(target=worker_entry_point, args=(
                    i, self.problem, shared_state, work_queues, termination_event, 
                    switch_to_bb_event, self.enable_heuristics
                ))
                workers.append(p)
                p.start()
            
            last_log_time = time.time()
            last_best_cost = shared_state.get_best_cost()
            is_min = self.problem.sense == "minimize"
            consecutive_gap_count = 0
            CONSECUTIVE_GAP_CHECKS_REQUIRED = 5
            
            while not termination_event.is_set():
                if self.timeout and (time.time() - start_time) > self.timeout:
                    if self.verbose: print(f"\n[Monitor]: CRITÉRIO DE PARADA ATINGIDO: Timeout de {self.timeout}s.")
                    termination_reason = f"Timeout ({self.timeout}s)"; break
                
                if self.stagnation_limit and shared_state.has_solution.value:
                    nodes_since_update = shared_state.nodes_processed.value - shared_state.get_last_update_node()
                    if nodes_since_update > self.stagnation_limit:
                        if self.verbose: print(f"\n[Monitor]: CRITÉRIO DE PARADA ATINGIDO: Estagnação ({self.stagnation_limit} nós).")
                        termination_reason = f"Stagnation ({self.stagnation_limit} nós)"; break
                
                if self.node_limit and shared_state.nodes_processed.value >= self.node_limit:
                    if self.verbose: print(f"\n[Monitor]: CRITÉRIO DE PARADA ATINGIDO: Limite de {self.node_limit} nós.")
                    termination_reason = f"Node Limit ({self.node_limit} nós)"; break

                new_best_cost = shared_state.get_best_cost()
                new_solution_found = new_best_cost != last_best_cost
                
                best_primal = new_best_cost
                worker_bounds = list(shared_state.worker_best_bounds.values())
                valid_worker_bounds = [b for b in worker_bounds if b is not None and ((is_min and b < float('inf')) or (not is_min and b > -float('inf')))]

                best_dual = float('-inf') if is_min else float('inf')
                if valid_worker_bounds:
                    best_dual = min(valid_worker_bounds) if is_min else max(valid_worker_bounds)
                
                gap = self._calculate_gap(best_primal, best_dual)

                if gap < 2:
                    consecutive_gap_count += 1
                else:
                    consecutive_gap_count = 0

                if consecutive_gap_count >= CONSECUTIVE_GAP_CHECKS_REQUIRED and not switch_to_bb_event.is_set():
                    if self.verbose: print(f"\n[Monitor]: Gap baixo detectado. Sinalizando para workers trocarem para Best-Bound...")
                    switch_to_bb_event.set()
                
                if gap < self.mip_gap_tolerance:
                    if self.verbose: print(f"\n[Monitor]: CRITÉRIO DE PARADA ATINGIDO: Gap de otimalidade ({gap:.4f}%) < tolerância.")
                    termination_reason = "Optimal (Gap Fechado)"; break
                
                if self.verbose and (new_solution_found or time.time() - last_log_time > 5):
                    nodes_done = shared_state.nodes_processed.value
                    idle_count = shared_state.get_idle_worker_count()
                    num_cuts = len(shared_state.get_cuts())
                    elapsed = time.time() - start_time
                    primal_str = f"{best_primal:.2f}" if shared_state.has_solution.value else "inf"
                    is_dual_valid = (is_min and best_dual > -float('inf')) or (not is_min and best_dual < float('inf'))
                    dual_str = f"{best_dual:.2f}" if is_dual_valid else ("-inf" if is_min else "inf")
                    gap_str = f"{gap:.2f}" if gap != float('inf') else "inf"
                    update_reason = "* New Solution" if new_solution_found else ""
                    print(f"{nodes_done:>8d}{idle_count:>6d}{num_cuts:>6d}{primal_str:>15s}{dual_str:>15s}{gap_str:>10s}{f'{elapsed:.1f}':>10s}{update_reason:>15s}")
                    last_log_time = time.time()
                    last_best_cost = new_best_cost
                
                if shared_state.get_idle_worker_count() == self.num_workers and shared_state.nodes_processed.value > 0:
                    time.sleep(0.5)
                    if shared_state.get_idle_worker_count() == self.num_workers:
                        if self.verbose: print(f"\n[Monitor]: Condição de término detectada. Todos os workers estão ociosos.")
                        termination_reason = "Optimal (Busca Concluída)"; break
                
                time.sleep(0.05)
        
        finally:
            if self.verbose:
                print("\n[Monitor]: Enviando sinal de término para todos os workers...")
            termination_event.set()
            for p in workers:
                p.join(timeout=2)
                if p.is_alive():
                    if self.verbose: print(f"[Monitor]: Worker {p.pid} não respondeu, forçando o encerramento.")
                    p.terminate()
            if self.verbose:
                print("[Monitor]: Todos os workers foram encerrados.")
        
        final_cost = shared_state.get_best_cost()
        final_solution = shared_state.get_best_solution()
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            print("\n" + "="*50)
            print("--- Resumo Final ---")
            print(f"Razão do término: {termination_reason}")
            print(f"Tempo total: {elapsed_time:.4f} segundos")
            if shared_state.has_solution.value:
                print(f"Melhor solução encontrada: {final_cost:.4f}")
            else:
                print("Nenhuma solução viável foi encontrada.")
            print("="*50)

        return final_cost, final_solution