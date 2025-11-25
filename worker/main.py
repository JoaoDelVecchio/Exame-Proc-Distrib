import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.repair import Repair
from pymoo.core.population import Population 
from pymoo.core.evaluator import Evaluator   

app = FastAPI()

class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        sum_x = X.sum(axis=1, keepdims=True)
        sum_x[sum_x == 0] = 1 
        return X / sum_x

class PortfolioProblemGA(ElementwiseProblem):
    def __init__(self, mu, cov, risk_free_rate=0.02, **kwargs):
        super().__init__(n_var=len(mu), n_obj=1, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        if x.ndim == 2:
            x = x.flatten()

        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        
        if exp_risk == 0:
            sharpe = 0
        else:
            sharpe = (exp_return - self.risk_free_rate) / exp_risk
        
        out["F"] = -sharpe
        out["sharpe"] = sharpe

class IslandState:
    def __init__(self):
        self.algorithm = None
        self.problem = None
        self.initialized = False

state = IslandState()

@app.post("/init")
def initialize():
    try:
        df = pd.read_csv("/app/data/portfolio_allocation.csv", parse_dates=True, index_col="date")
        
        returns = df.pct_change().dropna(how="all")
        mu = (1 + returns).prod() ** (252 / returns.count()) - 1
        cov = returns.cov() * 252
        
        mu_np = mu.to_numpy()
        cov_np = cov.to_numpy()

        state.problem = PortfolioProblemGA(mu_np, cov_np)
        
        state.algorithm = GA(
            pop_size=100,
            crossover=SBX(prob=0.9, eta=15, repair=PortfolioRepair()),
            mutation=PM(eta=20, repair=PortfolioRepair()),
            eliminate_duplicates=True
        )
        
        state.algorithm.setup(state.problem)
        state.initialized = True
        
        return {"status": "initialized", "assets": len(mu)}
    except Exception as e:
        print(f"Erro na inicializacao: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evolve")
def evolve(generations: int = 1):
    if not state.initialized:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    for _ in range(generations):
        state.algorithm.next()
        
    best_sharpe = state.algorithm.opt[0].get("sharpe")
    
    return {
        "status": "evolved", 
        "current_best_sharpe": float(best_sharpe)
    }

class MigrantData(BaseModel):
    genes: List[List[float]]

@app.get("/migrants")
def get_migrants():
    if not state.initialized:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    pop = state.algorithm.pop
    F = pop.get("F").flatten()
    sorted_indices = np.argsort(F)
    
    top_5_indices = sorted_indices[:5]
    top_5_genes = pop[top_5_indices].get("X")
    
    return {"genes": top_5_genes.tolist()}

@app.post("/migrants")
def receive_migrants(data: MigrantData):
    if not state.initialized:
        raise HTTPException(status_code=400, detail="Not initialized")

    new_genes_list = np.array(data.genes)
    if new_genes_list.ndim == 1:
        new_genes_list = new_genes_list.reshape(1, -1)
        
    migrant_pop = Population.new("X", new_genes_list)

    Evaluator().eval(state.problem, migrant_pop)

    main_pop = state.algorithm.pop
    F_main = main_pop.get("F").flatten()
    worst_indices = np.argsort(F_main)[-len(migrant_pop):]
    
    for i, idx in enumerate(worst_indices):
        main_pop[idx] = migrant_pop[i]
    
    return {"status": "migrants_integrated", "count": len(migrant_pop)}

@app.get("/status")
def status():
    if not state.initialized or state.algorithm.opt is None:
        return {"sharpe": 0.0}
    return {"sharpe": float(state.algorithm.opt[0].get("sharpe"))}