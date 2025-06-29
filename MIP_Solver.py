# -*- coding: utf-8 -*-

"""
SolverImperioMIP - Versão com Bandeira Heurística de Mergulho
Decreto de 24 de junho de 2025.
Dotando nosso solver com a audácia dos Bandeirantes.
"""

import gurobipy as gp
from gurobipy import GRB
import collections
import Location_Opt_Instance_Generator as Gen

Node = collections.namedtuple('Node', ['model', 'lower_bound'])


class SolverImperioMIP:
    # ... (O construtor e os outros métodos auxiliares permanecem os mesmos) ...
    # ... (Abaixo, incluí a classe completa para facilitar a cópia) ...
    def __init__(self, modelo_base: gp.Model, vars_inteiras: list, verbosidade: int = 1):
        """
        O Construtor, agora ciente de quais variáveis são inteiras.
        
        Args:
            modelo_base (gp.Model): O problema MIP a ser resolvido.
            vars_inteiras (list): Uma lista das variáveis Gurobi que devem ser inteiras.
            verbosidade (int): Nível de detalhe nas mensagens.
        """
        self.modelo_base = modelo_base
        self.vars_inteiras = vars_inteiras # Armazenamos a lista de nobres
        self.verbosidade = verbosidade
        # ... o resto do construtor permanece o mesmo ...
        self.modelo_base.setParam('OutputFlag', 0)
        self.melhor_solucao_inteira = None
        if self.modelo_base.ModelSense == GRB.MAXIMIZE:
            self.melhor_objetivo = -float('inf')
        else:
            self.melhor_objetivo = float('inf')
        self.num_nos_explorados = 0
        self.num_bandeiras_sucesso = 0

    # --- NOVO MÉTODO: A BANDEIRA HEURÍSTICA DE MERGULHO ---
    def _executar_bandeira_heuristica(self, modelo_inicial: gp.Model):
        """
        Executa uma expedição rápida (mergulho) em busca de uma solução inteira.
        É uma heurística "gananciosa", que não garante otimalidade, apenas busca viabilidade.
        """
        if self.verbosidade > 0:
            print(f"[BANDEIRA] Expedição iniciada a partir do nó #{self.num_nos_explorados}...")

        # O Bandeirante trabalha em seu próprio mapa, uma cópia, para não perturbar o Império.
        modelo_mergulho = modelo_inicial.copy()
        
        # Limitamos a jornada do Bandeirante para evitar que ele se perca.
        LIMITE_DE_MERGULHO = 20 

        for _ in range(LIMITE_DE_MERGULHO):
            modelo_mergulho.optimize()

            # Se o Bandeirante encontrou um beco sem saída (inviável)...
            if modelo_mergulho.status != GRB.OPTIMAL:
                if self.verbosidade > 0:
                    print("[BANDEIRA] A expedição encontrou um caminho inviável. Retornando.")
                return None # A expedição falhou.

            var_fracionaria = self._encontrar_primeira_variavel_fracionaria(modelo_mergulho)

            # Se não há mais variáveis fracionárias, a Bandeira encontrou a cidade do ouro!
            if var_fracionaria is None:
                if self.verbosidade > 0:
                    print(f"[BANDEIRA] SUCESSO! Solução inteira encontrada com objetivo {modelo_mergulho.objVal:.2f}.")
                self.num_bandeiras_sucesso += 1
                # Retorna o tesouro: o objetivo e a descrição da solução.
                return modelo_mergulho.objVal, {v.varName: v.x for v in modelo_mergulho.getVars()}

            # Se ainda há caminho a percorrer (variáveis fracionárias)...
            # O Bandeirante toma uma decisão rápida e pragmática: arredonda a variável.
            valor_fracionario = var_fracionaria.x
            valor_fixado = round(valor_fracionario)

            if self.verbosidade > 1: # Log mais detalhado
                 print(f"[BANDEIRA] Mergulhando: fixando {var_fracionaria.VarName} em {valor_fixado} (era {valor_fracionario:.2f})")

            # Adiciona a nova restrição ("travando" a variável) e continua o mergulho.
            var_a_fixar = modelo_mergulho.getVarByName(var_fracionaria.VarName)
            modelo_mergulho.addConstr(var_a_fixar == valor_fixado)
        
        if self.verbosidade > 0:
            print("[BANDEIRA] Limite de mergulho atingido. A expedição retorna sem sucesso imediato.")
        return None # A expedição foi muito longa e falhou.

    def solve(self):
        """ O método principal, com a Bandeira TEMPORARIAMENTE SUSPENSA. """
        if self.verbosidade > 0:
            print("--- Iniciando a Otimização do Império (MODO EXÉRCITO PRINCIPAL) ---")
            print("--- (Bandeirantes em descanso para observação do B&B) ---")

        fronteira = collections.deque()
        raiz = Node(model=self.modelo_base.copy(), lower_bound=-float('inf'))
        
        # --- DESPACHO DA PRIMEIRA BANDEIRA (TEMPORARIAMENTE SUSPENSO POR DECRETO IMPERIAL) ---
        # Para observarmos o Branch & Bound em sua forma pura, vamos
        # comentar a chamada da nossa heurística por um momento.
        #
        # resultado_bandeira = self._executar_bandeira_heuristica(raiz.model)
        # if resultado_bandeira:
        #     novo_objetivo, nova_solucao = resultado_bandeira
        #     self._atualizar_melhor_solucao(novo_objetivo, nova_solucao)
        # ---------------------------------------------------------------------------------

        fronteira.append(raiz)
        
        # O laço while agora fará todo o trabalho pesado.
        while fronteira:
            no_atual = fronteira.pop()
            self.num_nos_explorados += 1
            # ... o restante do método while continua exatamente o mesmo ...
            modelo_do_no = no_atual.model
            modelo_do_no.optimize()
            
            if modelo_do_no.status != gp.GRB.OPTIMAL:
                continue
            
            objetivo_do_no = modelo_do_no.objVal
            if self._deve_podar_por_limite(objetivo_do_no):
                continue
            
            var_fracionaria = self._encontrar_primeira_variavel_fracionaria(modelo_do_no)
            if var_fracionaria is None:
                self._atualizar_melhor_solucao(modelo_do_no.objVal, modelo_do_no)
            else:
                valor_fracionario = var_fracionaria.x
                modelo_ramo1 = modelo_do_no.copy()
                var_no_ramo1 = modelo_ramo1.getVarByName(var_fracionaria.VarName)
                modelo_ramo1.addConstr(var_no_ramo1 <= int(valor_fracionario))
                fronteira.append(Node(model=modelo_ramo1, lower_bound=objetivo_do_no))
                
                modelo_ramo2 = modelo_do_no.copy()
                var_no_ramo2 = modelo_ramo2.getVarByName(var_fracionaria.VarName)
                modelo_ramo2.addConstr(var_no_ramo2 >= int(valor_fracionario) + 1)
                fronteira.append(Node(model=modelo_ramo2, lower_bound=objetivo_do_no))
        
        if self.verbosidade > 0:
            self._imprimir_relatorio_final()
            
        return self.melhor_objetivo, self.melhor_solucao_inteira

    # O método de atualização agora pode ser chamado pelo solver principal ou pela heurística.
    # Modifiquei para aceitar um dicionário de solução diretamente.
    def _atualizar_melhor_solucao(self, novo_objetivo: float, solucao):
        # Se a solução for um modelo Gurobi, extraia o dicionário.
        if isinstance(solucao, gp.Model):
            solucao_dict = {v.varName: v.x for v in solucao.getVars()}
        # Se já for um dicionário, use-o diretamente.
        else:
            solucao_dict = solucao

        # A lógica de verificação e atualização permanece
        if self._deve_podar_por_limite(novo_objetivo):
            return

        self.melhor_objetivo = novo_objetivo
        self.melhor_solucao_inteira = solucao_dict
        
        if self.verbosidade > 0:
            print(f"Nova solução inteira encontrada! Objetivo: {novo_objetivo:.2f} (Nó #{self.num_nos_explorados})")

    # Demais métodos auxiliares (_deve_podar_por_limite, _encontrar_primeira_variavel_fracionaria, etc.)
    # permanecem os mesmos e são adicionados aqui para o código ser completo.
    def _deve_podar_por_limite(self, objetivo_do_no: float) -> bool:
        if self.modelo_base.ModelSense == GRB.MAXIMIZE:
            return objetivo_do_no <= self.melhor_objetivo
        else:
            return objetivo_do_no >= self.melhor_objetivo

    def _encontrar_primeira_variavel_fracionaria(self, modelo: gp.Model) -> gp.Var | None:
        """
        Percorre APENAS as variáveis que devem ser inteiras e retorna a primeira
        que possui valor fracionário. Uma lei justa e precisa.
        """
        TOLERANCIA = 1e-6
        # A lei agora é específica e só interroga os nobres!
        for var_info in self.vars_inteiras:
            # Precisamos pegar a variável correspondente no modelo do nó atual
            var_no_modelo_atual = modelo.getVarByName(var_info.VarName)
            if abs(var_no_modelo_atual.x - round(var_no_modelo_atual.x)) > TOLERANCIA:
                return var_no_modelo_atual
        return None

    def _imprimir_relatorio_final(self):
        print("\n--- Otimização Concluída ---")
        if self.num_bandeiras_sucesso > 0:
            print(f"Nossos Bandeirantes tiveram sucesso em {self.num_bandeiras_sucesso} expedição(ões)!")
        if self.melhor_solucao_inteira is not None:
            print(f"A melhor solução para o Império foi encontrada!")
            print(f"Valor da Função Objetivo: {self.melhor_objetivo:.4f}")
            print("Valores das Variáveis:")
            if self.melhor_solucao_inteira:
                for nome, valor in self.melhor_solucao_inteira.items():
                    print(f"  - {nome}: {valor:.2f}")
            else:
                print("  (Nenhuma variável na solução encontrada)")
        else:
            print("Infelizmente, não foi encontrada uma solução viável para o problema.")
        print(f"Total de nós explorados na árvore de decisão: {self.num_nos_explorados}")
        print("------------------------------")


if __name__ == '__main__':
    # --- DECRETO EXPERIMENTAL ---
    # 1. Mudamos o terreno com uma nova semente.
    # 2. Ordenamos ao Gurobi que desligue sua inteligência de pré-processamento.
    
    problema_desafio = Gen.criar_instancia_tsp(num_cidades=10)

    # Ordenamos ao motor que se comporte de forma mais "pura"
    problema_desafio.setParam("Presolve", 0)
    
    # ----------------------------

    # Coletamos as variáveis que devem ser inteiras (nossas variáveis 'y')
    # Uma forma mais robusta de fazer isso é usando o nome da variável.
    variaveis_y = [v for v in problema_desafio.getVars() if "abrir" in v.VarName]
    
    # E as entregamos ao nosso solver para que ele saiba quem vigiar.
    solver = SolverImperioMIP(problema_desafio, vars_inteiras=variaveis_y, verbosidade=1)
    
    # Mantemos os Bandeirantes em descanso para este teste crucial.
    objetivo, solucao = solver.solve()