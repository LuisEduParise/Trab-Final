# parallel_bb_solver.py (Corrigido para evitar Deadlock)
import time
import gurobipy as gp
from gurobipy import GRB
import multiprocessing as mp
from queue import Empty
from typing import List, Dict, Optional
import heapq
import traceback

from mip_problem import MIPProblem, Constraint
from tree_node import Node

# A classe NodeProcessor permanece idêntica.
# (Código omitido por brevidade)
class NodeProcessor:
    """Encapsula a lógica de processamento de um único nó do B&B."""
    def __init__(self, problem: MIPProblem):
        self.problem = problem
        self.vars_map = {var.name: var for var in self.problem.variables}
        self.integer_variables = [var.name for var in self.problem.variables if var.is_integer]
        self.binary_variables = {var.name for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1}
        self.tolerance = 1e-6
    def process_node(self, node: Node, best_bound_so_far: float, cut_pool: List[Constraint]) -> List:
        model = self._build_gurobi_model(node, cut_pool)
        model.optimize()
        results = []
        is_min = self.problem.sense == "minimize"
        objective_multiplier = 1.0 if is_min else -1.0
        if model.Status == GRB.OPTIMAL:
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
        if model.Status != GRB.OPTIMAL: return results
        lp_solution_value = model.ObjVal
        if (is_min and lp_solution_value >= best_bound_so_far) or \
           (not is_min and lp_solution_value <= best_bound_so_far):
            return results
        solution = {v.VarName: v.X for v in model.getVars()}
        fractional_var = self._select_most_fractional(solution)
        if fractional_var is None:
            results.append({'type': 'solution', 'value': lp_solution_value, 'solution': solution})
        else:
            val_to_branch = solution[fractional_var]
            new_depth = node.depth + 1
            parent_obj_for_heap = lp_solution_value * objective_multiplier
            node1 = Node(list(node.extra_bounds) + [(fractional_var, '<=', float(int(val_to_branch)))], parent_obj_for_heap, fractional_var, 'down', new_depth)
            node2 = Node(list(node.extra_bounds) + [(fractional_var, '>=', float(int(val_to_branch)) + 1)], parent_obj_for_heap, fractional_var, 'up', new_depth)
            results.append({'type': 'node', 'node': node1})
            results.append({'type': 'node', 'node': node2})
        return results
    def _build_gurobi_model(self, node: Node, cut_pool: List[Constraint]) -> gp.Model:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
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
        new_cuts = []
        for const in self.problem.constraints:
            if const.sense != '<=' or not all(v in self.binary_variables for v in const.coeffs.keys()): continue
            items_in_cover = {v: c for v, c in const.coeffs.items() if solution[v] > self.tolerance}
            cover_weight = sum(items_in_cover.values())
            if cover_weight <= const.rhs: continue
            minimal_cover = dict(items_in_cover)
            for var_name, coeff in items_in_cover.items():
                if cover_weight - coeff > const.rhs:
                    del minimal_cover[var_name]
                    cover_weight -= coeff
            if not minimal_cover: continue
            cut_rhs = len(minimal_cover) - 1
            if sum(solution[v] for v in minimal_cover.keys()) > cut_rhs + self.tolerance:
                new_cut = Constraint(coeffs={v: 1.0 for v in minimal_cover.keys()}, sense='<=', rhs=float(cut_rhs))
                if new_cut not in cut_pool and new_cut not in new_cuts:
                    new_cuts.append(new_cut)
        return new_cuts
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

# ==============================================================================
# FUNÇÃO DO TRABALHADOR (COM CORREÇÃO DE DEADLOCK)
# ==============================================================================
def worker_process_main_loop(task_queue, result_queue, shared_best_bound, problem, termination_event, shared_cut_pool_proxy):
    node_processor = NodeProcessor(problem)
    local_cut_pool = list(shared_cut_pool_proxy)
    dive_stack: List[Node] = []

    while not termination_event.is_set():
        node_to_process = None
        if dive_stack:
            node_to_process = dive_stack.pop()
        else:
            try:
                node_to_process = task_queue.get(timeout=0.1)
            except Empty:
                continue

        if len(local_cut_pool) < len(shared_cut_pool_proxy):
            local_cut_pool = list(shared_cut_pool_proxy)
        
        try:
            # --- CORREÇÃO CRÍTICA ---
            # O worker SEMPRE envia um resultado para o gerente, mesmo que seja uma lista vazia.
            # Isso sinaliza que a tarefa foi concluída e evita o deadlock.
            results = node_processor.process_node(node_to_process, shared_best_bound.value, local_cut_pool)
            
            nodes_for_manager = []
            if results:
                # A lógica de roteamento de resultados permanece a mesma...
                for res in results:
                    if res['type'] == 'node':
                        if not dive_stack:
                            dive_stack.append(res['node'])
                        else:
                            nodes_for_manager.append(res)
                    else:
                        nodes_for_manager.append(res)
            
            # ...mas a chamada para a fila agora está fora do 'if results'.
            result_queue.put(nodes_for_manager)

        except Exception as e:
            result_queue.put([{'type': 'error', 'error': e, 'traceback': traceback.format_exc()}])


# A classe ParallelBBSolver permanece idêntica à versão anterior.
# A correção está inteiramente contida na função do worker.
class ParallelBBSolver:
    """Implementa o Branch and Bound em paralelo com estratégia de Best-Bound."""
    def __init__(self, problem: MIPProblem, num_workers: Optional[int] = None):
        self.problem = problem
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.best_solution = None
        self.best_bound_value = float('inf') if self.problem.sense == 'minimize' else -float('inf')

    def solve(self):
        print("="*90)
        print(f"Iniciando B&B PARALELO para '{self.problem.name}' com {self.num_workers} workers.")
        print(f"Estratégia de Busca: Híbrida (Best-Bound Global + DFS Local)")
        print("="*90)

        start_time = time.time()
        
        with mp.Manager() as manager:
            task_queue = manager.Queue()
            result_queue = manager.Queue()
            shared_best_bound = manager.Value('d', self.best_bound_value)
            termination_event = manager.Event()
            shared_cut_pool = manager.list()

            node_heap = []
            is_min = self.problem.sense == 'minimize'
            objective_multiplier = 1.0 if is_min else -1.0

            root_node = Node(depth=0, parent_objective=float('-inf') * objective_multiplier)
            heapq.heappush(node_heap, root_node)
            
            workers = []
            for i in range(self.num_workers):
                p = mp.Process(target=worker_process_main_loop, args=(task_queue, result_queue, shared_best_bound, self.problem, termination_event, shared_cut_pool))
                p.start()
                workers.append(p)
            
            last_log_time = time.time()
            outstanding_tasks = 0

            while True:
                while outstanding_tasks < self.num_workers and node_heap:
                    best_node = heapq.heappop(node_heap)
                    task_queue.put(best_node)
                    outstanding_tasks += 1

                try:
                    results = result_queue.get(timeout=0.1)
                    
                    # Checa se o resultado não é um erro antes de decrementar o contador.
                    # A verificação original estava um pouco estranha, esta é mais direta.
                    # Se um erro ocorrer, tratamos o worker como "morto" e não esperamos mais por ele.
                    is_error = any(r.get('type') == 'error' for r in results)
                    if not is_error:
                        outstanding_tasks -= 1

                    for res in results:
                        if res['type'] == 'node':
                            heapq.heappush(node_heap, res['node'])
                        elif res['type'] == 'solution':
                            if (is_min and res['value'] < shared_best_bound.value) or \
                               (not is_min and res['value'] > shared_best_bound.value):
                                print(f"  -> Nova solução incumbente encontrada: {res['value']:.2f} (heap size: {len(node_heap)})")
                                shared_best_bound.value = res['value']
                                self.best_solution = res['solution']
                                self.best_bound_value = res['value']
                        elif res['type'] == 'cut':
                            if res['cut'] not in shared_cut_pool:
                                shared_cut_pool.append(res['cut'])
                        elif res['type'] == 'error':
                            print("\n!!! ERRO EM UM PROCESSO WORKER !!!")
                            print(res['traceback'])
                            termination_event.set() 
                
                except Empty:
                    if not node_heap and outstanding_tasks == 0:
                        break
                
                if termination_event.is_set():
                    break

                if time.time() - last_log_time > 2:
                    print(f"Update: {len(node_heap)} nós na fila global, {outstanding_tasks} em processo, {len(shared_cut_pool)} cortes, Melhor Bound: {self.best_bound_value:.2f}")
                    last_log_time = time.time()
            
            print("\nBusca terminada. Encerrando workers...")
            termination_event.set()
            for p in workers:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()

            elapsed_time = time.time() - start_time
            print("\n--- Resumo do Solver Paralelo ---")
            print(f"Status final: {'Optimal' if self.best_solution else 'Infeasible'}")
            print(f"  - Tempo total    : {elapsed_time:.4f} segundos")
            print(f"  - Workers usados : {self.num_workers}")
            if self.best_solution:
                print(f"Melhor solução encontrada:")
                print(f"  - Valor Objetivo: {self.best_bound_value:.4f}")