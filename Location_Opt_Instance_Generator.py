import random
import math
import gurobipy as gp

def criar_instancia_desafio(num_instalacoes=5, num_cidades=15, seed=2024):
    """
    Cria um modelo para o Problema de Localização com CAPACIDADE.
    Versão 3: O Desafio Final e Verdadeiro.
    """
    random.seed(seed)

    locais = range(num_instalacoes)
    cidades = range(num_cidades)

    # Parâmetros
    custos_fixos = {i: random.randint(8000, 15000) for i in locais}
    demandas = {j: random.randint(50, 150) for j in cidades}
    custos_transporte = {(i, j): random.randint(1, 10) for i in locais for j in cidades}
    
    # --- DECRETO DA CAPACIDADE LIMITADA ---
    # Cada instalação agora tem uma capacidade finita.
    # A capacidade é projetada para que uma única instalação não consiga suprir toda a demanda.
    demanda_total = sum(demandas.values())
    capacidades = {i: random.randint(int(demanda_total * 0.4), int(demanda_total * 0.7)) for i in locais}
    # ------------------------------------

    modelo = gp.Model("desafio_capacitado")

    y = modelo.addVars(locais, vtype=gp.GRB.BINARY, name="abrir")
    x = modelo.addVars(custos_transporte.keys(), vtype=gp.GRB.CONTINUOUS, name="enviar")

    modelo.setObjective(gp.quicksum(custos_fixos[i] * y[i] for i in locais) + 
                        gp.quicksum(custos_transporte[i, j] * x[i, j] for i, j in custos_transporte.keys()), 
                        gp.GRB.MINIMIZE)

    # Restrições
    modelo.addConstrs((x.sum('*', j) >= demandas[j] for j in cidades), name="demanda")
    modelo.addConstrs((x[i, j] <= demandas[j] * y[i] for i, j in custos_transporte.keys()), name="ligacao")
    
    # --- A NOVA LEI IMPERIAL ---
    modelo.addConstrs((x.sum(i, '*') <= capacidades[i] * y[i] for i in locais), name="capacidade")
    # -------------------------

    modelo.update()
    
    print("\n--- Instância de Desafio COM CAPACIDADE Criada ---")
    print(f"Problema com {num_instalacoes} instalações e {num_cidades} cidades.")
    print("A capacidade limitada forçará decisões complexas.")
    print("-------------------------------------------------")
    
    return modelo

def criar_instancia_tsp(num_cidades=10, seed=1889):
    """
    Cria um modelo Gurobi para o Problema do Caixeiro Viajante (PCV).
    O Desafio Supremo.
    
    Args:
        num_cidades (int): O número de províncias a serem visitadas.
        seed (int): Semente para reprodutibilidade, em homenagem à República.
    """
    random.seed(seed)

    # Gerar coordenadas aleatórias para as cidades em um mapa 2D
    pontos = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(num_cidades)}
    
    # Calcular a matriz de distâncias euclidianas
    distancias = {(i, j): math.sqrt((pontos[i][0] - pontos[j][0])**2 + (pontos[i][1] - pontos[j][1])**2)
                  for i in range(num_cidades) for j in range(num_cidades) if i != j}

    # --- Construção do Modelo ---
    modelo = gp.Model("problema_emissario_imperial")

    # Variáveis de decisão
    x = modelo.addVars(distancias.keys(), vtype=gp.GRB.BINARY, name="rota")
    u = modelo.addVars(num_cidades, vtype=gp.GRB.CONTINUOUS, name="ordem")

    # Função Objetivo
    modelo.setObjective(gp.quicksum(distancias[i, j] * x[i, j] for i, j in distancias.keys()), gp.GRB.MINIMIZE)

    # Restrições
    # 1. Partida Única
    modelo.addConstrs((x.sum(i, '*') == 1 for i in range(num_cidades)), name="partida")
    # 2. Chegada Única
    modelo.addConstrs((x.sum('*', j) == 1 for j in range(num_cidades)), name="chegada")
    
    # 3. Eliminação de Sub-rotas (MTZ)
    # u[i] - u[j] + n * x[i,j] <= n - 1  para i,j != 0
    for i in range(1, num_cidades):
        for j in range(1, num_cidades):
            if i != j:
                modelo.addConstr(u[i] - u[j] + num_cidades * x[i, j] <= num_cidades - 1, name=f"MTZ_{i}_{j}")
    
    modelo.update()
    
    print(f"\n--- Instância de Desafio PCV ({num_cidades} cidades) Criada ---")
    
    return modelo

# O if __name__ == '__main__' pode permanecer como no último teste,
# apenas garantindo que esta nova função seja chamada.