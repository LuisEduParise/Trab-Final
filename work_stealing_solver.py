# work_stealing_solver.py
import multiprocessing as mp
import time
import random
import heapq
from queue import Empty
from typing import List, Dict, Optional
import traceback
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

from mip_problem import MIPProblem, Constraint
from tree_node import Node
from parallel_utils import SharedState
from multiprocessing.synchronize import Event as MpEvent

class NodeProcessor:
    def __init__(self, problem: MIPProblem):
        self.problem = problem
        self.vars_map = {var.name: var for var in self.problem.variables}
        self.integer_variables = [var.name for var in self.problem.variables if var.is_integer]
        self.binary_variables = {var.name for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1}
        self.tolerance = 1e-6
        self.pseudocosts = defaultdict(lambda: {'down': {'sum_degrad': 0.0, 'count': 0}, 'up': {'sum_degrad': 0.0, 'count': 0}})
        self.warmup_nodes = 500
        self.local_node_count = 0

    def run_feasibility_pump(self) -> Optional[Dict]:
        print("[Feasibility Pump]: Iniciando busca por solução inicial...")
        root_node = Node()
        model = self._build_gurobi_model(root_node, [])
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
                print(f"[Feasibility Pump]: Solução viável encontrada na iteração {i+1} com custo {cost:.2f}!")
                return {'type': 'solution', 'value': cost, 'solution': rounded_solution}
            dist_model = self._build_distance_model(rounded_solution)
            dist_model.optimize()
            if dist_model.Status != GRB.OPTIMAL: break
            model = dist_model
        print("[Feasibility Pump]: Heurística encerrada sem solução.")
        return None

    def _is_solution_feasible(self, solution: Dict[str, float]) -> bool:
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
        best_var, best_score = None, -1.0
        fractional_vars = [v for v in self.integer_variables if abs(solution[v] - round(solution[v])) > self.tolerance]
        if not fractional_vars: return None
        for var_name in fractional_vars:
            val = solution[var_name]; f_down = val - int(val); f_up = 1.0 - f_down
            down_stats, up_stats = self.pseudocosts[var_name]['down'], self.pseudocosts[var_name]['up']
            pc_down = (down_stats['sum_degrad'] / down_stats['count']) if down_stats['count'] > 0 else 1.0
            pc_up = (up_stats['sum_degrad'] / up_stats['count']) if up_stats['count'] > 0 else 1.0
            score = (f_down * pc_down) + (f_up * pc_up)
            if score > best_score: best_score, best_var = score, var_name
        return best_var

    def _select_most_fractional(self, solution: Dict[str, float]) -> Optional[str]:
        best_var, max_frac_dist = None, -1
        for var_name in self.integer_variables:
            val = solution[var_name]
            if abs(val - round(val)) > self.tolerance:
                frac_dist_from_half = abs(abs(val - int(val)) - 0.5)
                if best_var is None or frac_dist_from_half < max_frac_dist:
                    max_frac_dist, best_var = frac_dist_from_half, var_name
        return best_var

    def _select_branching_variable(self, solution: Dict[str, float]) -> Optional[str]:
        if self.local_node_count <= self.warmup_nodes:
            return self._select_most_fractional(solution)
        else:
            return self._select_by_pseudocost(solution) or self._select_most_fractional(solution)

    def process_node(self, node: Node, best_bound_so_far: float, cut_pool: List[Constraint]) -> List:
        self.local_node_count += 1
        model = self._build_gurobi_model(node, cut_pool)
        model.optimize()
        
        results, is_min = [], self.problem.sense == "minimize"
        
        if model.Status == GRB.OPTIMAL:
            self._update_pseudocosts(node, model.ObjVal)
            for _ in range(5):
                solution = {v.VarName: v.X for v in model.getVars()}
                if self._select_most_fractional(solution) is None: break
                new_cuts = self._generate_knapsack_cuts(solution, cut_pool)
                if not new_cuts: break
                results.extend([{'type': 'cut', 'cut': c} for c in new_cuts])
                for cut in new_cuts:
                    expr = gp.LinExpr([(c, model.getVarByName(v)) for v, c in cut.coeffs.items()]); model.addConstr(expr <= cut.rhs)
                model.optimize()

        if model.Status != GRB.OPTIMAL:
            return results

        lp_solution_value = model.ObjVal
        
        if (is_min and lp_solution_value >= best_bound_so_far) or (not is_min and lp_solution_value <= best_bound_so_far):
            return results

        solution = {v.VarName: v.X for v in model.getVars()}
        fractional_var = self._select_branching_variable(solution)

        if fractional_var is None:
            results.append({'type': 'solution', 'value': lp_solution_value, 'solution': solution})
        else:
            val_to_branch = solution[fractional_var]
            new_depth = node.depth + 1
            
            objective_for_heap = lp_solution_value if is_min else -lp_solution_value
            
            node1 = Node(
                lp_bound=lp_solution_value,
                parent_objective=objective_for_heap,
                extra_bounds=list(node.extra_bounds) + [(fractional_var, '<=', float(int(val_to_branch)))], 
                branch_variable=fractional_var, 
                branch_direction='down', 
                depth=new_depth
            )
            node2 = Node(
                lp_bound=lp_solution_value,
                parent_objective=objective_for_heap,
                extra_bounds=list(node.extra_bounds) + [(fractional_var, '>=', float(int(val_to_branch)) + 1)], 
                branch_variable=fractional_var, 
                branch_direction='up', 
                depth=new_depth
            )
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
            expr = gp.LinExpr([(c, gurobi_vars[v]) for v, c in cut.coeffs.items()]); model.addConstr(expr <= cut.rhs)
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
                if cover_weight - coeff > const.rhs: del minimal_cover[var_name]; cover_weight -= coeff
            if not minimal_cover: continue
            cut_rhs = len(minimal_cover) - 1
            if sum(solution[v] for v in minimal_cover.keys()) > cut_rhs + self.tolerance:
                new_cut = Constraint(coeffs={v: 1.0 for v in minimal_cover.keys()}, sense='<=', rhs=float(cut_rhs))
                if new_cut not in cut_pool: new_cuts.append(new_cut)
        return new_cuts

class ParallelWorker:
    def __init__(self, worker_id: int, problem: MIPProblem, shared_state: SharedState, work_queues: List[mp.Queue], termination_event: MpEvent):
        self.worker_id, self.problem, self.shared_state, self.work_queues, self.termination_event = worker_id, problem, shared_state, work_queues, termination_event
        self.my_queue = self.work_queues[self.worker_id]
        self.node_processor = NodeProcessor(self.problem)
        self.local_heap: List[Node] = []
        self.is_idle = False
        self.num_workers = len(work_queues)
        print(f"[Worker {self.worker_id}]: Iniciado.")

    def run(self):
        if not self.shared_state.has_solution.value:
            Node.strategy = "dfs"
            if self.worker_id == 0:
                print(f"[Worker 0]: Todos os workers iniciando com a estratégia DFS.")
        else:
            Node.strategy = "best_bound"

        if self.worker_id == 0:
            initial_solution = self.node_processor.run_feasibility_pump()
            if initial_solution: 
                self.shared_state.update_best_solution(initial_solution['value'], initial_solution['solution'])
            
            is_min = self.problem.sense == 'minimize'
            initial_bound = float('-inf') if is_min else float('inf')
            initial_objective_for_heap = float('-inf')
            
            root_node = Node(
                lp_bound=initial_bound, 
                parent_objective=initial_objective_for_heap, 
                depth=0
            )
            heapq.heappush(self.local_heap, root_node)
            
        while not self.termination_event.is_set():
            if self.shared_state.has_solution.value and Node.strategy != "best_bound":
                print(f"[Worker {self.worker_id}]: Primeira solução encontrada! Trocando para estratégia Best-Bound.")
                Node.strategy = "best_bound"
                heapq.heapify(self.local_heap)

            self._handle_incoming_messages()
            if self.local_heap:
                if self.is_idle: self.shared_state.decrement_idle_worker_count(); self.is_idle = False
                
                # Garante que o bound reportado é sempre o melhor do heap local.
                best_local_bound_to_report = None
                if self.problem.sense == 'minimize':
                    min_bound = float('inf')
                    for node in self.local_heap:
                        if node.lp_bound is not None and node.lp_bound < min_bound:
                            min_bound = node.lp_bound
                    if min_bound < float('inf'):
                        best_local_bound_to_report = min_bound
                else: # Maximize
                    max_bound = -float('inf')
                    for node in self.local_heap:
                        if node.lp_bound is not None and node.lp_bound > max_bound:
                            max_bound = node.lp_bound
                    if max_bound > -float('inf'):
                        best_local_bound_to_report = max_bound

                if best_local_bound_to_report is not None:
                    self.shared_state.update_worker_best_bound(self.worker_id, best_local_bound_to_report)
                
                node_to_process = heapq.heappop(self.local_heap)
                current_global_cuts = self.shared_state.get_cuts()
                results = self.node_processor.process_node(node_to_process, self.shared_state.get_best_cost(), current_global_cuts)
                self.shared_state.increment_nodes_processed()
                new_cuts_found = []
                for res in results:
                    if res['type'] == 'node': heapq.heappush(self.local_heap, res['node'])
                    elif res['type'] == 'solution':
                        if self.shared_state.update_best_solution(res['value'], res['solution']):
                            pass
                    elif res['type'] == 'cut': new_cuts_found.append(res['cut'])
                if new_cuts_found: self.shared_state.add_cuts(new_cuts_found)
            else:
                if not self.is_idle:
                    self.shared_state.increment_idle_worker_count(); self.is_idle = True
                    invalid_bound = float('inf') if self.problem.sense == 'minimize' else -float('inf')
                    self.shared_state.update_worker_best_bound(self.worker_id, invalid_bound)
                self._request_work()
                time.sleep(0.1 + random.uniform(0, 0.2))
                
        print(f"[Worker {self.worker_id}]: Sinal de término recebido. Encerrando.")
    
    def _handle_incoming_messages(self):
        while not self.my_queue.empty():
            try:
                message = self.my_queue.get_nowait()
                msg_type = message.get('type')
                if msg_type == 'STEAL_REQUEST': self._handle_steal_request(message['from_id'])
                elif msg_type == 'WORK_RESPONSE': self._handle_work_response(message['nodes'])
            except Empty: return

    def _handle_steal_request(self, requester_id: int):
        if len(self.local_heap) > 1:
            num_to_share = len(self.local_heap) // 2
            if Node.strategy == 'dfs':
                nodes_to_share = heapq.nsmallest(num_to_share, self.local_heap)
            else:
                nodes_to_share = heapq.nlargest(num_to_share, self.local_heap)
                
            self.local_heap = [node for node in self.local_heap if node not in nodes_to_share]
            heapq.heapify(self.local_heap)
            if nodes_to_share: self.work_queues[requester_id].put({'type': 'WORK_RESPONSE', 'nodes': nodes_to_share})

    def _handle_work_response(self, nodes: List[Node]):
        if nodes:
            for node in nodes: heapq.heappush(self.local_heap, node)

    def _request_work(self):
        target_id = random.choice([i for i in range(self.num_workers) if i != self.worker_id])
        self.work_queues[target_id].put({'type': 'STEAL_REQUEST', 'from_id': self.worker_id})

def worker_entry_point(worker_id: int, problem: MIPProblem, shared_state: SharedState, work_queues: List[mp.Queue], termination_event: MpEvent):
    try:
        worker = ParallelWorker(worker_id, problem, shared_state, work_queues, termination_event)
        worker.run()
    except Exception as e:
        print(f"[Worker {worker_id}]: ERRO FATAL - {e}"); traceback.print_exc()

class WorkStealingSolver:
    def __init__(self, problem: MIPProblem, num_workers: Optional[int] = None, 
                 timeout: Optional[float] = None, stagnation_limit: Optional[int] = None, 
                 mip_gap_tolerance: float = 1e-4):
        self.problem = problem
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.timeout = timeout
        self.stagnation_limit = stagnation_limit
        self.mip_gap_tolerance = mip_gap_tolerance

    def _calculate_gap(self, best_primal, best_dual) -> float:
        is_min = self.problem.sense == "minimize"
        if (is_min and best_primal == float('inf')) or (not is_min and best_primal == -float('inf')) or \
           (is_min and best_dual == -float('inf')) or (not is_min and best_dual == float('inf')):
            return float('inf')
        if abs(best_primal) < 1e-9:
            return float('inf') if abs(best_primal - best_dual) > 1e-9 else 0.0
        return 100.0 * abs(best_primal - best_dual) / (abs(best_primal) + 1e-9)

    def solve(self):
        start_time = time.time()
        termination_reason = "Desconhecido"
        
        print("="*100)
        print(f"Iniciando Solver com Work Stealing para '{self.problem.name}' com {self.num_workers} workers.")
        print(f"{'Nodes':>8s}{'Idle':>6s}{'Cuts':>6s}{'BestInt':>15s}{'BestBound':>15s}{'Gap(%)':>10s}{'Time(s)':>10s}{'Update':>15s}")
        print("-"*100)
        
        workers = []
        try:
            shared_state = SharedState(self.num_workers, self.problem.sense)
            termination_event = mp.Event()
            work_queues = [mp.Queue() for _ in range(self.num_workers)]
            
            for i in range(self.num_workers):
                p = mp.Process(target=worker_entry_point, args=(i, self.problem, shared_state, work_queues, termination_event))
                workers.append(p)
                p.start()
            
            last_log_time = time.time()
            last_best_cost = shared_state.get_best_cost()
            
            is_min = self.problem.sense == "minimize"
            global_best_dual = float('-inf') if is_min else float('inf')
            
            while not termination_event.is_set():
                if self.timeout and (time.time() - start_time) > self.timeout:
                    print(f"\n[Monitor]: CRITÉRIO DE PARADA ATINGIDO: Timeout de {self.timeout}s.")
                    termination_reason = f"Timeout ({self.timeout}s)"; break
                
                if self.stagnation_limit and shared_state.has_solution.value:
                    nodes_since_update = shared_state.nodes_processed.value - shared_state.get_last_update_node()
                    if nodes_since_update > self.stagnation_limit:
                        print(f"\n[Monitor]: CRITÉRIO DE PARADA ATINGIDO: Estagnação ({self.stagnation_limit} nós).")
                        termination_reason = f"Stagnation ({self.stagnation_limit} nós)"; break
                
                new_best_cost = shared_state.get_best_cost()
                new_solution_found = new_best_cost != last_best_cost
                
                best_primal = new_best_cost
                worker_bounds = list(shared_state.worker_best_bounds.values())
                
                valid_worker_bounds = [b for b in worker_bounds if b is not None and ((is_min and b < float('inf')) or (not is_min and b > -float('inf')))]

                if valid_worker_bounds:
                    current_best_from_workers = min(valid_worker_bounds) if is_min else max(valid_worker_bounds)
                    
                    if is_min:
                        global_best_dual = max(global_best_dual, current_best_from_workers)
                    else:
                        global_best_dual = min(global_best_dual, current_best_from_workers)

                best_dual = global_best_dual
                
                gap = self._calculate_gap(best_primal, best_dual)
                
                if gap < self.mip_gap_tolerance:
                    print(f"\n[Monitor]: CRITÉRIO DE PARADA ATINGIDO: Gap de otimalidade ({gap:.4f}%) < tolerância.")
                    termination_reason = "Optimal (Gap Fechado)"; break
                
                if new_solution_found or time.time() - last_log_time > 5:
                    nodes_done = shared_state.nodes_processed.value
                    idle_count = shared_state.get_idle_worker_count()
                    num_cuts = len(shared_state.get_cuts())
                    elapsed = time.time() - start_time
                    
                    primal_str = f"{best_primal:.2f}" if shared_state.has_solution.value else "inf"
                    is_dual_valid = (self.problem.sense == 'minimize' and best_dual > -float('inf')) or \
                                    (self.problem.sense == 'maximize' and best_dual < float('inf'))
                    dual_str = f"{best_dual:.2f}" if is_dual_valid else ("-inf" if self.problem.sense == 'minimize' else "inf")
                    
                    gap_str, time_str = f"{gap:.2f}" if gap != float('inf') else "inf", f"{elapsed:.1f}"
                    update_reason = "* New Solution" if new_solution_found else ""
                    
                    print(f"{nodes_done:>8d}{idle_count:>6d}{num_cuts:>6d}{primal_str:>15s}{dual_str:>15s}{gap_str:>10s}{time_str:>10s}{update_reason:>15s}")
                    last_log_time = time.time()
                    last_best_cost = new_best_cost
                
                idle_count = shared_state.get_idle_worker_count()
                if idle_count == self.num_workers and shared_state.nodes_processed.value > 0:
                    time.sleep(0.5) 
                    if shared_state.get_idle_worker_count() == self.num_workers:
                        print(f"\n[Monitor]: Condição de término detectada. Todos os workers estão ociosos.")
                        termination_reason = "Optimal (Busca Concluída)"; break
                
                time.sleep(0.05)
        
        finally:
            print("\n[Monitor]: Enviando sinal de término para todos os workers...")
            termination_event.set()
            for p in workers:
                p.join(timeout=2)
                if p.is_alive():
                    print(f"[Monitor]: Worker {p.pid} não respondeu, forçando o encerramento.")
                    p.terminate()
            print("[Monitor]: Todos os workers foram encerrados.")
        
        final_cost = shared_state.get_best_cost()
        final_solution = shared_state.get_best_solution()
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
