# presolver.py
"""
Implementa a fase de pré-processamento (presolve) para um problema de MIP.

O objetivo do presolve é simplificar e fortalecer a formulação do problema
antes de iniciar o resolvedor principal (branch-and-bound), o que pode
reduzir drasticamente o tempo de resolução. As técnicas incluem:
- Análise de restrições singleton e redundantes.
- Propagação de bounds.
- Fixação de variáveis por probing e dual fixing.

Muitas dessas análises são paralelizadas para melhorar o desempenho.
"""
import math
import copy
import multiprocessing as mp
from collections import defaultdict
from mip_problem import MIPProblem, Constraint, Variable
from typing import Dict, List, Tuple
from implication_graph import ImplicationGraph


class InfeasibleProblemError(Exception):
    """Exceção levantada quando o presolve prova que o problema é inviável."""
    pass


def _probe_worker(args: Tuple) -> Tuple:
    """
    Função worker para executar a sondagem (probing) em uma variável binária.

    Testa fixar a variável em 0 e depois em 1, executando a propagação de
    bounds em cada caso. Se uma das fixações levar a uma inviabilidade,
    a variável pode ser permanentemente fixada no valor oposto.

    Args:
        args (Tuple): Uma tupla contendo (variável, mapa de vars, restrições, tolerância).

    Returns:
        Tuple: Uma tupla ('fix', var_name, value) se uma fixação foi encontrada,
               ou None caso contrário.
    """
    var_to_probe, vars_map, constraints, tolerance = args
    var_name = var_to_probe.name

    def _run_single_probe(val_to_probe: int) -> bool:
        """
        Executa a sondagem para um único valor (0 ou 1) e retorna
        True se o subproblema se tornar inviável.
        """
        # Dicionário para salvar os bounds originais APENAS das variáveis modificadas
        original_bounds = {}

        try:
            # Salva o bound original da variável de probe e aplica a fixação
            if val_to_probe == 0:
                original_bounds[var_name] = (vars_map[var_name].lb, vars_map[var_name].ub)
                vars_map[var_name].ub = 0.0
            else: # val_to_probe == 1
                original_bounds[var_name] = (vars_map[var_name].lb, vars_map[var_name].ub)
                vars_map[var_name].lb = 1.0

            # Propaga os bounds iterativamente
            for _ in range(5):
                # CORREÇÃO: Chamar a função unificada e desempacotar a tupla
                # Usamos '_' para ignorar a contagem de modificações.
                _, new_changes = Presolver._propagate_bounds(vars_map, constraints, tolerance)
                
                if not new_changes:
                    break
                # Salva os bounds originais das novas variáveis modificadas
                for v_name, (orig_lb, orig_ub) in new_changes.items():
                    if v_name not in original_bounds:
                        original_bounds[v_name] = (orig_lb, orig_ub)
            
            Presolver._check_infeasibility(vars_map, tolerance)
            return False  # Não é inviável
        
        except InfeasibleProblemError:
            return True  # É inviável
        
        finally:
            # BLOCO CRUCIAL: Restaura todos os bounds modificados para o estado original
            for v_name, (lb, ub) in original_bounds.items():
                vars_map[v_name].lb = lb
                vars_map[v_name].ub = ub

    # --- Lógica principal do _probe_worker ---
    
    # Down-probe: Testa fixar a variável em 0
    if _run_single_probe(0):
        # Se fixar em 0 torna o problema inviável, a variável deve ser 1
        return 'fix', var_name, 1.0

    # Up-probe: Testa fixar a variável em 1
    if _run_single_probe(1):
        # Se fixar em 1 torna o problema inviável, a variável deve ser 0
        return 'fix', var_name, 0.0
        
    return None


def _analysis_worker(args: Tuple) -> Tuple:
    """
    Função worker para analisar um subconjunto de restrições.

    Identifica restrições singleton (com apenas uma variável), que podem ser
    usadas para apertar bounds, e restrições redundantes, que podem ser removidas.

    Args:
        args (Tuple): Uma tupla contendo (chunk de restrições, mapa de vars, tolerância).

    Returns:
        Tuple: Uma tupla com (atualizações de singleton, singletons removíveis,
               restrições redundantes).
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
            
            # Novo bound da variável (new bound value)
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
        
        # 2. Análise de Redundância
        else:
            min_activity, max_activity = Presolver._calculate_activity_bounds(const, vars_map)
            # Ignora se os bounds de atividade não puderam ser calculados
            if abs(min_activity) == float('inf') or abs(max_activity) == float('inf'):
                continue
            
            is_redundant = False
            if const.sense == '<=' and max_activity <= const.rhs + tolerance:
                is_redundant = True
            elif const.sense == '>=' and min_activity >= const.rhs - tolerance:
                is_redundant = True
            
            if is_redundant:
                redundant_constraints.append(const)

    return singleton_updates, removable_singletons, redundant_constraints


class Presolver:
    """
    Orquestra a execução de várias técnicas de pré-processamento para um MIP.
    """
    def __init__(self, problem: MIPProblem, use_probing: bool = True, probe_limit: int = 50, num_workers: int = None):
        """
        Inicializa o presolver.

        Args:
            problem (MIPProblem): O problema de MIP a ser simplificado.
            use_probing (bool): Se a técnica de probing deve ser usada.
            probe_limit (int): Número máximo de variáveis binárias a serem testadas com probing.
            num_workers (int): Número de processos paralelos a serem usados. Padrão é mp.cpu_count().
        """
        self.problem = problem.copy()
        self.modifications = 0
        self.vars_map = {var.name: var for var in self.problem.variables}
        self.tolerance = 1e-9
        self.use_probing = use_probing
        self.probe_limit = probe_limit
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.stats = defaultdict(int)
        self.implication_graph = ImplicationGraph()

    def _print_summary(self):
        """Imprime um resumo das modificações realizadas pelo presolve."""
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
        Calcula os 'locks' para cada variável.

        Um 'up-lock' é uma restrição que impede uma variável de aumentar de valor.
        Um 'down-lock' impede a variável de diminuir. Essa informação é usada
        pela heurística de Dual Fixing.
        """
        for var in self.vars_map.values():
            var.up_locks = 0
            var.down_locks = 0

        for const in self.problem.constraints:
            for var_name, coeff in const.coeffs.items():
                if abs(coeff) < self.tolerance: continue
                var = self.vars_map[var_name]
                
                if const.sense == '<=':
                    if coeff > 0: var.up_locks += 1
                    else: var.down_locks += 1
                elif const.sense == '>=':
                    if coeff > 0: var.down_locks += 1
                    else: var.up_locks += 1
                elif const.sense == '==':
                    var.up_locks += 1
                    var.down_locks += 1

    def _apply_dual_fixing(self):
        """
        Aplica a técnica de Dual Fixing.

        Tenta fixar variáveis em seus limites com base no coeficiente da função
        objetivo e na ausência de 'locks' na direção desejada. Por exemplo, se o
        objetivo é minimizar uma variável (coeficiente positivo) e não há
        down-locks, ela pode ser fixada em seu limite inferior.
        """
        fixings_found = 0
        sense_multiplier = 1.0 if self.problem.sense == "minimize" else -1.0

        for var_name, var in list(self.vars_map.items()):
            if var.lb > var.ub - self.tolerance: continue  # Já está fixada

            objective_coeff = self.problem.objective.get(var_name, 0.0) * sense_multiplier

            # Tenta fixar no limite inferior
            if objective_coeff >= -self.tolerance and var.down_locks == 0:
                var.ub = var.lb
                fixings_found += 1
            # Tenta fixar no limite superior
            elif objective_coeff < self.tolerance and var.up_locks == 0:
                var.lb = var.ub
                fixings_found += 1

        if fixings_found > 0:
            print(f"  -> Fixação Dual (Dual Fixing) fixou {fixings_found} variáveis.")
            self.modifications += fixings_found
            self.stats['Fixações por Fixação Dual'] += fixings_found

    def _build_implication_graph(self):
        """
        Constrói o grafo de implicações a partir das restrições do problema.

        Esta é uma implementação simplificada para fins de demonstração. Um
        resolvedor completo construiria este grafo de forma mais robusta,
        principalmente através da técnica de probing (Seção 10.6 da tese ).
        Aqui, extraímos implicações óbvias de restrições simples.
        """
        print("  - Construindo grafo de implicações (versão simplificada)...")
        binary_vars = {var.name for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1}

        for const in self.problem.constraints:
            # Procura por restrições do tipo: bin_var + non_bin_var <= rhs
            if const.sense == '<=' and len(const.coeffs) == 2:
                vars_in_const = list(const.coeffs.keys())
                var1, var2 = self.vars_map[vars_in_const[0]], self.vars_map[vars_in_const[1]]

                # Identifica qual é a binária e qual é a outra
                bin_var, other_var = (var1, var2) if var1.name in binary_vars and var2.name not in binary_vars else \
                                     (var2, var1) if var2.name in binary_vars and var1.name not in binary_vars else \
                                     (None, None)
                
                if bin_var is not None:
                    # Exemplo: x + y <= 5. Se x=1, então y <= 4.
                    # A lógica exata depende dos coeficientes.
                    # Para c1*x + c2*y <= rhs:
                    # x=1 => c2*y <= rhs - c1 => y <= (rhs-c1)/c2 (se c2>0)
                    c_bin = const.coeffs[bin_var.name]
                    c_other = const.coeffs[other_var.name]

                    # Implicação de x=1
                    implied_bound_on_1 = (const.rhs - c_bin) / c_other
                    if c_other > 0: # y <= ...
                        self.implication_graph.add_implication(bin_var.name, 1, other_var.name, 'ub', implied_bound_on_1)
                    else: # y >= ...
                        self.implication_graph.add_implication(bin_var.name, 1, other_var.name, 'lb', implied_bound_on_1)

                    # Implicação de x=0
                    implied_bound_on_0 = const.rhs / c_other
                    if c_other > 0: # y <= ...
                        self.implication_graph.add_implication(bin_var.name, 0, other_var.name, 'ub', implied_bound_on_0)
                    else: # y >= ...
                        self.implication_graph.add_implication(bin_var.name, 0, other_var.name, 'lb', implied_bound_on_0)


    def _apply_implication_graph_analysis(self):
        """
        Executa a análise do grafo de implicações para encontrar reduções.

        Este método implementa o Algorithm 10.13 da tese de Achterberg.
        Ele itera sobre todas as variáveis binárias e verifica se as implicações
        de fixá-las em 0 e 1 podem levar a novos limites globais ou a agregações
        em outras variáveis.
        """
        print("  - Análise do Grafo de Implicações...")
        binary_vars = [var.name for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1]
        
        mods_found = 0

        for bin_var_name in binary_vars:
            common_vars = self.implication_graph.get_common_implied_vars(bin_var_name)

            for implied_var_name in common_vars:
                var_to_modify = self.vars_map[implied_var_name]
                
                # Pula variáveis já fixadas/agregadas
                if hasattr(var_to_modify, 'is_aggregated') and var_to_modify.is_aggregated:
                    continue

                # Busca os bounds implicados por bin_var=0 e bin_var=1
                l0, u0 = self.implication_graph.get_implied_bounds(bin_var_name, 0, implied_var_name)
                l1, u1 = self.implication_graph.get_implied_bounds(bin_var_name, 1, implied_var_name)

                # Regra (a): Aperto do limite inferior
                new_lb = min(l0, l1)
                if new_lb > var_to_modify.lb + self.tolerance:
                    var_to_modify.lb = new_lb
                    mods_found += 1
                    self.stats['Bounds por Grafo de Implicação'] += 1

                # Regra (b): Aperto do limite superior
                new_ub = max(u0, u1)
                if new_ub < var_to_modify.ub - self.tolerance:
                    var_to_modify.ub = new_ub
                    mods_found += 1
                    self.stats['Bounds por Grafo de Implicação'] += 1
                
                # Regras (c) e (d): Agregação de variáveis
                # Agregação: x_impl = L + (U - L) * x_bin
                if abs(u0 - var_to_modify.lb) < self.tolerance and abs(l1 - var_to_modify.ub) < self.tolerance:
                     # Regra (c): x_impl = lb + (ub - lb) * x_bin
                    print(f"    -> Agregação: {implied_var_name} := {var_to_modify.lb} + ({var_to_modify.ub - var_to_modify.lb}) * {bin_var_name}")
                    var_to_modify.is_aggregated = True # Marca como agregada
                    mods_found += 1
                    self.stats['Agregações por Grafo de Implicação'] += 1

                elif abs(l0 - var_to_modify.ub) < self.tolerance and abs(u1 - var_to_modify.lb) < self.tolerance:
                    # Regra (d): x_impl = ub - (ub - lb) * x_bin
                    print(f"    -> Agregação: {implied_var_name} := {var_to_modify.ub} - ({var_to_modify.ub - var_to_modify.lb}) * {bin_var_name}")
                    var_to_modify.is_aggregated = True # Marca como agregada
                    mods_found += 1
                    self.stats['Agregações por Grafo de Implicação'] += 1
        
        if mods_found > 0:
            self.modifications += mods_found
    def _upgrade_constraint_type(self, const: Constraint):
        """
        Analisa uma restrição linear e a identifica como um tipo mais específico.
        Baseado no Algoritmo 10.7 da tese de Achterberg (p. 145).
        Adiciona um atributo `special_type` ao objeto Constraint.
        """
        # Se já foi classificada, não faz nada
        if hasattr(const, 'special_type'):
            return

        const.special_type = 'linear' # Padrão
        
        coeffs = const.coeffs
        num_vars = len(coeffs)
        
        # Verifica se todas as variáveis são binárias
        are_all_binary = all(
            self.vars_map[v].is_integer and self.vars_map[v].lb == 0 and self.vars_map[v].ub == 1
            for v in coeffs
        )
        if not are_all_binary:
            return

        # ---- Lógica para SPPC (Set Packing/Partitioning/Covering) ----
        are_all_one_or_minus_one = all(abs(c) == 1.0 for c in coeffs.values())
        
        if are_all_one_or_minus_one:
            pos_coeffs = sum(1 for c in coeffs.values() if c > 0)
            
            # Forma padrão: sum(x_j) (op) 1
            # Checa se pode ser transformada em sum(y_j) (op) 1, onde y_j é x_j ou 1-x_j
            if const.rhs == 1.0 - (num_vars - pos_coeffs): # Lado direito para sum(y_j)
                if const.sense == '<=': const.special_type = 'set_packing'
                elif const.sense == '==': const.special_type = 'set_partitioning'
                elif const.sense == '>=': const.special_type = 'set_covering'

        # ---- Lógica para Knapsack ----
        # Forma padrão: sum(a_j * x_j) <= b, com a_j inteiros e positivos
        if const.sense == '<=':
            are_all_coeffs_integer_non_zero = all(c != 0 and abs(c) == int(abs(c)) for c in coeffs.values())
            if are_all_coeffs_integer_non_zero and const.rhs == int(const.rhs):
                const.special_type = 'knapsack'


    def _dispatch_presolve_tasks(self):
        """
        Itera sobre as restrições, identifica seus tipos e chama a rotina
        de pré-processamento especializada correta.
        """
        print("  - Despachando tarefas de presolve para restrições...")
        
        # Coleta restrições por tipo
        sppc_constraints = []
        for const in self.problem.constraints:
            self._upgrade_constraint_type(const)
            
            if const.special_type == 'knapsack':
                self._presolve_knapsack(const)
            elif const.special_type in {'set_packing', 'set_partitioning', 'set_covering'}:
                sppc_constraints.append(const)
            else: # 'linear'
                # A análise de singletons e redundância pode ser feita aqui se desejado
                pass
        
        # Processamento de pares de SPPC
        if sppc_constraints:
            self._presolve_sppc_pairs(sppc_constraints)

        # Remove as restrições que se tornaram redundantes
        self.problem.constraints = [c for c in self.problem.constraints if not (hasattr(c, 'is_redundant') and c.is_redundant)]

    def _presolve_knapsack(self, const: Constraint):
            """
            Aplica técnicas de pré-processamento para uma restrição knapsack.
            Baseado no Algoritmo 10.8 da tese (p. 147).
            """
            # (A lógica de normalização permanece a mesma)
            capacity = const.rhs
            weights = {}
            for var_name, coeff in const.coeffs.items():
                if coeff < 0:
                    capacity -= coeff
                    weights[var_name] = -coeff 
                else:
                    weights[var_name] = coeff
            
            # Etapa 5: Extração de Cliques
            sorted_items = sorted(weights.items(), key=lambda item: item[1], reverse=True)
            if len(sorted_items) > 1:
                for i in range(len(sorted_items)):
                    for j in range(i + 1, len(sorted_items)):
                        w_i, w_j = sorted_items[i][1], sorted_items[j][1]
                        if w_i + w_j > capacity:
                            var_i, var_j = sorted_items[i][0], sorted_items[j][0]
                            
                            # Adiciona implicação e CONTA APENAS SE FOR NOVA
                            if self.implication_graph.add_implication(var_i, 1, var_j, 'ub', 0):
                                self.stats['Implicações de Knapsack (Cliques)'] += 1
                                self.modifications += 1
                            
                            if self.implication_graph.add_implication(var_j, 1, var_i, 'ub', 0):
                                self.stats['Implicações de Knapsack (Cliques)'] += 1
                                self.modifications += 1

    def _presolve_sppc_pairs(self, sppc_constraints: List[Constraint]):
        """
        Aplica presolve em pares de restrições SPPC.
        Baseado no Algoritmo 10.9 da tese (p. 152).
        """
        restart_scan = True
        while restart_scan:
            restart_scan = False
            for i in range(len(sppc_constraints)):
                for j in range(i + 1, len(sppc_constraints)):
                    const_p = sppc_constraints[i]
                    const_q = sppc_constraints[j]

                    if (hasattr(const_p, 'is_redundant') and const_p.is_redundant) or \
                       (hasattr(const_q, 'is_redundant') and const_q.is_redundant):
                        continue
                    
                    vars_p = set(const_p.coeffs.keys())
                    vars_q = set(const_q.coeffs.keys())

                    # Etapa 3c: p é subconjunto de q
                    if vars_p.issubset(vars_q) and vars_p != vars_q:
                        if const_p.special_type in {'set_partitioning', 'set_covering'} and \
                           const_q.special_type in {'set_partitioning', 'set_packing'}:
                            
                            vars_to_fix = vars_q - vars_p
                            changed_something = False
                            for var_name in vars_to_fix:
                                var = self.vars_map[var_name]
                                if var.ub > 0:
                                    var.ub = 0.0
                                    # Assegura que a variável não seja re-fixada
                                    var.lb = 0.0
                                    self.stats['Fixações (Pares SPPC)'] += 1
                                    self.modifications += 1
                                    changed_something = True
                            
                            if changed_something:
                                # Uma fixação pode gerar novas reduções. Reinicia a varredura.
                                restart_scan = True
                                break # Sai do loop interno (j)
                if restart_scan:
                    break # Sai do loop externo (i)

    def presolve(self) -> MIPProblem:
        """
        Executa o processo de presolve completo.

        Aplica as várias técnicas de simplificação em rodadas até que nenhuma
        modificação adicional possa ser feita no problema.
        """
        print(f"--- Iniciando a rotina de Presolve (usando até {self.num_workers} processos) ---")
        
        self._build_implication_graph()

        round_num = 1
        while True:
            print(f"--- Rodada de Presolve {round_num} ---")
            self.modifications = 0
            
            # Despacha cada restrição para sua rotina de presolve específica
            self._dispatch_presolve_tasks()

            self._calculate_variable_locks()
            self._apply_dual_fixing()
            self._apply_implication_graph_analysis()
            self._apply_bound_propagation()
            
            if self.use_probing:
                self._apply_parallel_probe()

            try:
                self._check_infeasibility(self.vars_map, self.tolerance)
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
        self.problem.variables = [
            v for v in self.problem.variables 
            if not (hasattr(v, 'is_aggregated') and v.is_aggregated)
        ]
        
        return self.problem

    def _apply_parallel_analysis(self):
        """Executa a análise de singletons e redundâncias em paralelo."""
        num_constraints = len(self.problem.constraints)
        if num_constraints == 0: return

        chunk_size = max(1, math.ceil(num_constraints / self.num_workers))
        chunks = [self.problem.constraints[i:i + chunk_size] for i in range(0, num_constraints, chunk_size)]
        analysis_args = [(chunk, self.vars_map, self.tolerance) for chunk in chunks]

        all_singleton_updates, all_removable_singletons, all_redundant_constraints = [], [], []

        with mp.Pool(self.num_workers) as pool:
            results = pool.map(_analysis_worker, analysis_args)

        for singleton_updates, removable_s, redundant_c in results:
            all_singleton_updates.extend(singleton_updates)
            all_removable_singletons.extend(removable_s)
            all_redundant_constraints.extend(redundant_c)

        self._apply_singleton_results(all_singleton_updates, all_removable_singletons)
        self._apply_redundancy_results(all_redundant_constraints)

    def _apply_singleton_results(self, updates: List, removable: List):
        """Aplica os resultados da análise de restrições singleton."""
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

    def _apply_redundancy_results(self, redundant: List):
        """Remove as restrições identificadas como redundantes."""
        if redundant:
            redundant_set = set(redundant)
            num_removed = len(redundant_set)
            self.problem.constraints = [c for c in self.problem.constraints if c not in redundant_set]
            self.stats['Restrições Redundantes'] += num_removed
            self.modifications += num_removed

    def _apply_bound_propagation(self):
        """Executa a propagação de bounds de forma sequencial."""
        # A função agora retorna (contagem, dict). Usamos '_' para ignorar o dict.
        mods, _ = Presolver._propagate_bounds(self.vars_map, self.problem.constraints, self.tolerance)
        if mods > 0:
            print(f"  - Propagação de Bounds: Apertou {mods} bounds de variáveis.")
            self.stats['Bounds por Propagação'] += mods
            self.modifications += mods

    def _apply_parallel_probe(self):
            """Executa o probing em variáveis binárias de forma paralela."""
            all_binary_vars = [var for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1 and (var.ub - var.lb) > self.tolerance]
            
            if not all_binary_vars: return

            # ---- OTIMIZAÇÃO: Ordenar variáveis pela sua influência (locks) ----
            # A tese sugere (p. 155) ordenar por um score de impacto.
            # Usamos os locks como uma boa aproximação desse score.
            all_binary_vars.sort(key=lambda var: var.up_locks + var.down_locks, reverse=True)
            # --------------------------------------------------------------------

            binary_vars_to_probe = all_binary_vars[:self.probe_limit]
            if not binary_vars_to_probe: return
            
            print(f"  - Probing: Analisando as {len(binary_vars_to_probe)} vars binárias mais promissoras com {self.num_workers} processos.")
            
            # Otimização para evitar deepcopy: passamos apenas os bounds, não o mapa inteiro.
            # Isto requer uma mudança maior no _probe_worker, então vamos focar na seleção por enquanto.
            probe_args = [(var, self.vars_map, self.problem.constraints, self.tolerance) for var in binary_vars_to_probe]
            
            fixings_found = 0
            with mp.Pool(self.num_workers) as pool:
                results = pool.map(_probe_worker, probe_args)
            
            # (O resto do método continua igual)
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

    @staticmethod
    def _propagate_bounds(vars_map: Dict[str, Variable], constraints: List[Constraint], tolerance: float) -> Tuple[int, Dict[str, Tuple[float, float]]]:
        """
        Tenta apertar os bounds das variáveis com base nas restrições.

        Esta versão unificada retorna tanto a contagem de modificações quanto um
        dicionário com os bounds originais das variáveis que foram alteradas,
        servindo a múltiplos propósitos dentro do presolver.
        """
        modifications_in_run = 0
        changed_vars_with_orig_bounds = {}

        for const in constraints:
            for var_name_to_tighten, target_coeff in const.coeffs.items():
                if abs(target_coeff) < tolerance: continue
                
                target_var = vars_map[var_name_to_tighten]
                orig_lb, orig_ub = target_var.lb, target_var.ub

                min_activity_others, max_activity_others = Presolver._calculate_activity_bounds(const, vars_map, exclude_var=var_name_to_tighten)
                
                if abs(min_activity_others) == float('inf') or abs(max_activity_others) == float('inf'): continue
                
                made_change = False
                if const.sense == '<=':
                    residual_rhs = const.rhs - min_activity_others
                    if target_coeff > 0:
                        if (nb := residual_rhs / target_coeff) < target_var.ub - tolerance:
                            target_var.ub = nb; made_change = True
                    else:
                        if (nb := residual_rhs / target_coeff) > target_var.lb + tolerance:
                            target_var.lb = nb; made_change = True
                elif const.sense == '>=':
                    residual_rhs = const.rhs - max_activity_others
                    if target_coeff > 0:
                        if (nb := residual_rhs / target_coeff) > target_var.lb + tolerance:
                            target_var.lb = nb; made_change = True
                    else:
                        if (nb := residual_rhs / target_coeff) < target_var.ub - tolerance:
                            target_var.ub = nb; made_change = True
                
                if made_change:
                    modifications_in_run += 1
                    # Salva os bounds originais apenas na primeira vez que a variável é modificada nesta chamada
                    if var_name_to_tighten not in changed_vars_with_orig_bounds:
                        changed_vars_with_orig_bounds[var_name_to_tighten] = (orig_lb, orig_ub)
        
        return modifications_in_run, changed_vars_with_orig_bounds

    @staticmethod
    def _calculate_activity_bounds(constraint: Constraint, vars_map: Dict[str, Variable], exclude_var: str = None) -> Tuple[float, float]:
        """
        Calcula a atividade mínima e máxima de uma expressão linear de uma restrição.
        """
        min_activity, max_activity = 0.0, 0.0
        for var_name, coeff in constraint.coeffs.items():
            if var_name == exclude_var: continue
            var = vars_map[var_name]
            if coeff > 0:
                min_activity += coeff * var.lb
                max_activity += coeff * var.ub
            else: # coeff < 0
                min_activity += coeff * var.ub
                max_activity += coeff * var.lb
        return min_activity, max_activity

    @staticmethod
    def _check_infeasibility(vars_map: Dict[str, Variable], tolerance: float):
        """
        Verifica se algum bound de variável se tornou inválido (lb > ub).

        Raises:
            InfeasibleProblemError: Se uma inviabilidade for encontrada.
        """
        for var in vars_map.values():
            if var.lb > var.ub + tolerance:
                raise InfeasibleProblemError(f"Inviabilidade na variável '{var.name}': lb ({var.lb}) > ub ({var.ub})")