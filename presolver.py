# presolver.py
import math
import copy
import multiprocessing as mp
from collections import defaultdict
from mip_problem import MIPProblem, Constraint

class InfeasibleProblemError(Exception):
    pass

# --- WORKER PARA PROBING (JÁ EXISTENTE) ---
def _probe_worker(args):
    """
    Executa a sondagem (up e down probe) para uma única variável binária.
    """
    var, vars_map, constraints, tolerance = args
    var_name = var.name
    
    # Down-probe
    probe_down_map = copy.deepcopy(vars_map)
    probe_down_map[var_name].ub = 0.0
    try:
        for _ in range(5):
            if Presolver._propagate_bounds(probe_down_map, constraints, tolerance) == 0: break
        Presolver._check_infeasibility(probe_down_map, tolerance)
    except InfeasibleProblemError:
        return ('fix', var_name, 1.0)

    # Up-probe
    probe_up_map = copy.deepcopy(vars_map)
    probe_up_map[var_name].lb = 1.0
    try:
        for _ in range(5):
            if Presolver._propagate_bounds(probe_up_map, constraints, tolerance) == 0: break
        Presolver._check_infeasibility(probe_up_map, tolerance)
    except InfeasibleProblemError:
        return ('fix', var_name, 0.0)
        
    return None

# --- NOVO: WORKER PARA ANÁLISE DE SINGLETONS E REDUNDÂNCIA ---
def _analysis_worker(args):
    """
    Analisa um subconjunto de restrições para identificar singletons e redundâncias.
    Projetado para ser executado em um processo paralelo.
    """
    constraints_chunk, vars_map, tolerance = args
    
    singleton_updates = []
    removable_singletons = []
    redundant_constraints = []

    for const in constraints_chunk:
        # 1. Análise de Singletons
        if len(const.coeffs) == 1:
            var_name, coeff = list(const.coeffs.items())[0]
            if abs(coeff) < tolerance:
                removable_singletons.append(const)
                continue
            
            nbv = const.rhs / coeff
            if const.sense == '<=':
                if coeff > 0: singleton_updates.append((var_name, 'ub', nbv))
                else: singleton_updates.append((var_name, 'lb', nbv))
            elif const.sense == '>=':
                if coeff > 0: singleton_updates.append((var_name, 'lb', nbv))
                else: singleton_updates.append((var_name, 'ub', nbv))
            elif const.sense == '==':
                singleton_updates.append((var_name, 'lb', nbv))
                singleton_updates.append((var_name, 'ub', nbv))
            removable_singletons.append(const)
        
        # 2. Análise de Redundância (só executa se não for singleton)
        else:
            min_activity, max_activity = Presolver._calculate_activity_bounds(const, vars_map)
            if abs(min_activity) == float('inf') or abs(max_activity) == float('inf'):
                continue
            
            is_redundant = False
            if const.sense == '<=' and max_activity <= const.rhs + tolerance: is_redundant = True
            elif const.sense == '>=' and min_activity >= const.rhs - tolerance: is_redundant = True
            
            if is_redundant:
                redundant_constraints.append(const)

    return singleton_updates, removable_singletons, redundant_constraints

class Presolver:
    def __init__(self, problem: MIPProblem, use_probing: bool = True, probe_limit: int = 500, num_workers: int = None):
        self.problem = problem.copy()
        self.modifications = 0
        self.vars_map = {var.name: var for var in self.problem.variables}
        self.tolerance = 1e-9
        self.use_probing = use_probing
        self.probe_limit = probe_limit
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.stats = defaultdict(int)

    def _print_summary(self):
        print("\n--- Resumo do Presolve ---")
        if not self.stats:
            print("Nenhuma modificação foi realizada.")
            return
        total_modifications = sum(self.stats.values())
        print(f"Total de modificações: {total_modifications}")
        print("-" * 45)
        print(f"{'Técnica':<25s} | {'Modificações':>15s}")
        print("-" * 25 + "|-" + "-" * 18)
        for tech, count in sorted(self.stats.items()):
            print(f"{tech:<25s} | {count:>16d}")
        print("-" * 45)

    def _calculate_variable_locks(self):
        """
        Calcula os contadores up_locks e down_locks para cada variável.
        Esta informação é fundamental para a heurística de Dual Fixing. 
        """
        # Primeiro, reseta os contadores de todas as variáveis
        for var in self.vars_map.values():
            var.up_locks = 0
            var.down_locks = 0

        # Itera sobre cada restrição para calcular os locks
        for const in self.problem.constraints:
            for var_name, coeff in const.coeffs.items():
                if abs(coeff) < self.tolerance:
                    continue

                var = self.vars_map[var_name]
                
                if const.sense == '<=':
                    if coeff > 0:
                        var.up_locks += 1
                    else: # coeff < 0
                        var.down_locks += 1
                elif const.sense == '>=':
                    if coeff > 0:
                        var.down_locks += 1
                    else: # coeff < 0
                        var.up_locks += 1
                elif const.sense == '==':
                    # Uma igualdade restringe o movimento em ambas as direções
                    var.up_locks += 1
                    var.down_locks += 1

    def _apply_dual_fixing(self):
        """
        Aplica a técnica de Dual Fixing (Seção 10.8 de Achterberg).
        Usa os locks e os coeficientes da função objetivo para fixar variáveis.
        """
        fixings_found = 0
        # Assumimos que o sentido é "minimize". A lógica é invertida para "maximize".
        sense_multiplier = 1.0 if self.problem.sense == "minimize" else -1.0

        # Iteramos sobre as variáveis para checar se podem ser fixadas
        for var_name, var in list(self.vars_map.items()):
            # Pula variáveis que já estão fixadas
            if var.lb > var.ub - self.tolerance:
                continue

            objective_coeff = self.problem.objective.get(var_name, 0.0) * sense_multiplier

            # Caso 1: Fixar no limite inferior (lower bound)
            # Se o objetivo quer minimizar a variável (coeff >= 0) e não há
            # restrições impedindo-a de diminuir (down_locks == 0).
            if objective_coeff >= -self.tolerance and var.down_locks == 0:
                # Fixa a variável ao tornar seu limite superior igual ao inferior
                var.ub = var.lb
                fixings_found += 1
                
            # Caso 2: Fixar no limite superior (upper bound)
            # Se o objetivo quer maximizar a variável (coeff < 0) e não há
            # restrições impedindo-a de aumentar (up_locks == 0).
            elif objective_coeff < self.tolerance and var.up_locks == 0:
                # Fixa a variável ao tornar seu limite inferior igual ao superior
                var.lb = var.ub
                fixings_found += 1

        if fixings_found > 0:
            print(f"  -> Fixação Dual (Dual Fixing) fixou {fixings_found} variáveis.")
            self.modifications += fixings_found
            self.stats['Fixações por Fixação Dual'] += fixings_found

    def presolve(self) -> MIPProblem:
        print(f"--- Iniciando a rotina de Presolve (usando até {self.num_workers} processos) ---")
        round_num = 1
        while True:
            self.modifications = 0

            self._calculate_variable_locks()

            self._apply_dual_fixing()
            
            # ATUALIZADO: Executa análise de singletons e redundância em paralelo
            self._apply_parallel_analysis()
            
            # A propagação de bounds continua sequencial devido à sua natureza
            self._apply_bound_propagation()
            
            if self.use_probing:
                self._apply_parallel_probe()

            try:
                Presolver._check_infeasibility(self.vars_map, self.tolerance)
            except InfeasibleProblemError:
                print("\n!!! Presolve detectou que o problema é INVIÁVEL !!!")
                self._print_summary()
                raise

            print(f"Rodada {round_num} de Presolve completada. Modificações: {self.modifications}")
            if self.modifications == 0:
                break
            round_num += 1
            
        print("--- Presolve finalizado ---")
        self._print_summary()
        return self.problem

    def _apply_parallel_analysis(self):
        """Executa a análise de singletons e redundâncias em paralelo."""
        num_constraints = len(self.problem.constraints)
        if num_constraints == 0: return

        # Divide a lista de restrições em pedaços (chunks) para os workers
        chunk_size = max(1, math.ceil(num_constraints / self.num_workers))
        chunks = [self.problem.constraints[i:i + chunk_size] for i in range(0, num_constraints, chunk_size)]
        analysis_args = [(chunk, self.vars_map, self.tolerance) for chunk in chunks]

        all_singleton_updates = []
        all_removable_singletons = []
        all_redundant_constraints = []

        # Executa a análise em paralelo
        with mp.Pool(self.num_workers) as pool:
            results = pool.map(_analysis_worker, analysis_args)

        # Agrega os resultados de todos os workers
        for singleton_updates, removable_s, redundant_c in results:
            all_singleton_updates.extend(singleton_updates)
            all_removable_singletons.extend(removable_s)
            all_redundant_constraints.extend(redundant_c)

        # Aplica as modificações de forma sequencial para garantir consistência
        self._apply_singleton_results(all_singleton_updates, all_removable_singletons)
        self._apply_redundancy_results(all_redundant_constraints)


    def _apply_singleton_results(self, updates, removable):
        if updates:
            num_bounds_changed = 0
            for var_name, bound_type, value in updates:
                if bound_type == 'lb' and value > self.vars_map[var_name].lb + self.tolerance:
                    self.vars_map[var_name].lb = value
                    num_bounds_changed += 1
                elif bound_type == 'ub' and value < self.vars_map[var_name].ub - self.tolerance:
                    self.vars_map[var_name].ub = value
                    num_bounds_changed += 1
            if num_bounds_changed > 0:
                self.stats['Bounds por Singletons'] += num_bounds_changed
                self.modifications += num_bounds_changed
        
        if removable:
            removable_set = set(removable)
            num_removed = len(removable_set)
            self.problem.constraints = [c for c in self.problem.constraints if c not in removable_set]
            self.stats['Restrições (Singleton)'] += num_removed
            self.modifications += num_removed

    def _apply_redundancy_results(self, redundant):
        if redundant:
            redundant_set = set(redundant)
            num_removed = len(redundant_set)
            self.problem.constraints = [c for c in self.problem.constraints if c not in redundant_set]
            self.stats['Restrições Redundantes'] += num_removed
            self.modifications += num_removed

    def _apply_bound_propagation(self):
        mods = Presolver._propagate_bounds(self.vars_map, self.problem.constraints, self.tolerance)
        if mods > 0:
            print(f"  - Presolve/Propagation: Apertou {mods} bounds de variáveis (execução sequencial).")
            self.stats['Bounds por Propagação'] += mods
            self.modifications += mods

    def _apply_parallel_probe(self):
        all_binary_vars = [var for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1 and (var.ub - var.lb) > self.tolerance]
        binary_vars_to_probe = all_binary_vars[:self.probe_limit]
        if not binary_vars_to_probe: return
        
        print(f"  - Presolve/Probing: Analisando {len(binary_vars_to_probe)} vars binárias com {self.num_workers} processos.")
        probe_args = [(var, self.vars_map, self.problem.constraints, self.tolerance) for var in binary_vars_to_probe]
        
        fixings_found = 0
        with mp.Pool(self.num_workers) as pool:
            results = pool.map(_probe_worker, probe_args)
        
        for res in results:
            if res is not None:
                _, var_name, value = res
                if value == 1.0 and self.vars_map[var_name].lb < 1.0:
                    self.vars_map[var_name].lb = 1.0
                    fixings_found += 1
                elif value == 0.0 and self.vars_map[var_name].ub > 0.0:
                    self.vars_map[var_name].ub = 0.0
                    fixings_found += 1

        if fixings_found > 0:
            print(f"  -> Probing (paralelo) fixou {fixings_found} variáveis.")
            self.modifications += fixings_found
            self.stats['Fixações por Probing'] += fixings_found

    # --- Métodos estáticos (sem alterações) ---
    @staticmethod
    def _propagate_bounds(vars_map, constraints, tolerance):
        modifications_in_run = 0
        for const in constraints:
            for var_name_to_tighten, target_coeff in const.coeffs.items():
                if abs(target_coeff) < tolerance: continue
                target_var = vars_map[var_name_to_tighten]
                min_activity_others, max_activity_others = Presolver._calculate_activity_bounds(const, vars_map, exclude_var=var_name_to_tighten)
                if abs(min_activity_others) == float('inf') or abs(max_activity_others) == float('inf'): continue
                if const.sense == '<=':
                    residual_rhs = const.rhs - min_activity_others
                    if target_coeff > 0:
                        if (nb := residual_rhs / target_coeff) < target_var.ub - tolerance: target_var.ub = nb; modifications_in_run += 1
                    else:
                        if (nb := residual_rhs / target_coeff) > target_var.lb + tolerance: target_var.lb = nb; modifications_in_run += 1
                elif const.sense == '>=':
                    residual_rhs = const.rhs - max_activity_others
                    if target_coeff > 0:
                        if (nb := residual_rhs / target_coeff) > target_var.lb + tolerance: target_var.lb = nb; modifications_in_run += 1
                    else:
                        if (nb := residual_rhs / target_coeff) < target_var.ub - tolerance: target_var.ub = nb; modifications_in_run += 1
        return modifications_in_run

    @staticmethod
    def _calculate_activity_bounds(constraint, vars_map, exclude_var=None):
        min_activity, max_activity = 0, 0
        for var_name, coeff in constraint.coeffs.items():
            if var_name == exclude_var: continue
            var = vars_map[var_name]
            if coeff > 0:
                min_activity += coeff * var.lb
                max_activity += coeff * var.ub
            else:
                min_activity += coeff * var.ub
                # --- CORREÇÃO APLICADA AQUI ---
                max_activity += coeff * var.lb
        return min_activity, max_activity

    @staticmethod
    def _check_infeasibility(vars_map, tolerance):
        for var in vars_map.values():
            if var.lb > var.ub + tolerance:
                raise InfeasibleProblemError(f"Inviabilidade na variável '{var.name}': lb ({var.lb}) > ub ({var.ub})")
