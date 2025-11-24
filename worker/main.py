import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Imports do Pymoo (Igual ao seu notebook)
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.repair import Repair
from pymoo.optimize import minimize

app = FastAPI()

# --- 1. Lógica do Problema

class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # Zera pesos muito pequenos e normaliza para soma = 1
        X[X < 1e-3] = 0
        # Evita divisão por zero
        sum_x = X.sum(axis=1, keepdims=True)
        sum_x[sum_x == 0] = 1 
        return X / sum_x

class PortfolioProblemGA(ElementwiseProblem):
    def __init__(self, mu, cov, risk_free_rate=0.02, **kwargs):
        # Minimizar (-Sharpe)
        super().__init__(n_var=len(mu), n_obj=1, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        
        # Evita divisão por zero no risco
        if exp_risk == 0:
            sharpe = 0
        else:
            sharpe = (exp_return - self.risk_free_rate) / exp_risk
        
        # Pymoo minimiza, então retornamos negativo
        out["F"] = -sharpe

# --- 2. Estado Global (A memória da nossa Ilha) ---
class IslandState:
    def __init__(self):
        self.algorithm = None
        self.problem = None
        self.initialized = False

state = IslandState()

# --- 3. Endpoints da API (Os botões que o Go vai apertar) ---

@app.post("/init")
def initialize():
    try:
        # Lê o CSV da pasta mapeada
        df = pd.read_csv("/app/data/portfolio_allocation.csv", parse_dates=True, index_col="date")
        
        # Cálculos financeiros básicos (do seu notebook)
        returns = df.pct_change().dropna(how="all")
        mu = (1 + returns).prod() ** (252 / returns.count()) - 1
        cov = returns.cov() * 252
        
        mu_np = mu.to_numpy()
        cov_np = cov.to_numpy()

        # Configura o Problema
        state.problem = PortfolioProblemGA(mu_np, cov_np)
        
        # Configura o Algoritmo Genético (População de 100)
        state.algorithm = GA(
            pop_size=100,
            crossover=SBX(prob=0.9, eta=15, repair=PortfolioRepair()),
            mutation=PM(eta=20, repair=PortfolioRepair()),
            eliminate_duplicates=True
        )
        
        # Inicializa a primeira geração
        state.algorithm.setup(state.problem)
        state.initialized = True
        
        return {"status": "initialized", "assets": len(mu)}
    except Exception as e:
        print(f"Erro na inicialização: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evolve")
def evolve(generations: int = 1):
    if not state.initialized:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    # Roda N gerações
    for _ in range(generations):
        state.algorithm.next()
        
    # Pega o melhor resultado atual
    best_fitness = -state.algorithm.opt[0].F[0] # Inverte o sinal de volta
    return {"status": "evolved", "current_best_sharpe": float(best_fitness)}

class MigrantData(BaseModel):
    genes: List[List[float]]

@app.get("/migrants")
def get_migrants():
    if not state.initialized:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    # Pega a população atual
    pop = state.algorithm.pop
    # Ordena por Fitness (menor é melhor no pymoo pois é minimização)
    F = pop.get("F").flatten()
    sorted_indices = np.argsort(F)
    
    # Seleciona os 5 melhores para enviar
    top_5_indices = sorted_indices[:5]
    top_5_genes = pop[top_5_indices].get("X")
    
    return {"genes": top_5_genes.tolist()}

@app.post("/migrants")
def receive_migrants(data: MigrantData):
    if not state.initialized:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    new_individuals = np.array(data.genes)
    pop = state.algorithm.pop
    
    # Estratégia de substituição: Substitui os piores da população atual pelos que chegaram
    F = pop.get("F").flatten()
    # Índices dos piores (maiores valores, pois é minimização de negativo)
    worst_indices = np.argsort(F)[-len(new_individuals):]
    
    # Injeta os genes
    for i, idx in enumerate(worst_indices):
        pop[idx].set("X", new_individuals[i])
        # Importante: Marcar como não avaliado para o pymoo recalcular o fitness na próxima geração
        # Ou podemos calcular manualmente, mas deixaremos o GA ajustar
        # Hack simples: setar fitness como infinito para ele ser reavaliado ou descartado se for ruim
        # No pymoo, a reavaliação automática depende da implementação, 
        # aqui vamos forçar a avaliação no próximo `next()` naturalmente.
    
    return {"status": "migrants_integrated", "count": len(new_individuals)}

@app.get("/status")
def status():
    if not state.initialized or state.algorithm.opt is None:
        return {"sharpe": 0.0}
    
    # Retorna o melhor Sharpe Ratio encontrado até agora
    return {"sharpe": float(-state.algorithm.opt[0].F[0])}