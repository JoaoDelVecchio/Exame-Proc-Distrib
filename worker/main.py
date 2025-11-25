import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from islands import *
from constants import *

app = FastAPI()

state = IslandState()

@app.post("/init")
def initialize():
    try:
        df = pd.read_csv("/app/data/portfolio_allocation.csv", parse_dates=True, index_col="date")
        
        returns = df.pct_change().dropna(how="all")
        mu = (1 + returns).prod() ** (BUSSINESS_DAYS / returns.count()) - 1
        cov = returns.cov() * BUSSINESS_DAYS
        
        mu_np = mu.to_numpy()
        cov_np = cov.to_numpy()

        state.initialize(mu_np, cov_np)

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
    
    top_indices = sorted_indices[:MIGRANTS]
    top_genes = pop[top_indices].get("X")
    
    return {"genes": top_genes.tolist()}

@app.post("/migrants")
def receive_migrants(data: MigrantData):
    if not state.initialized:
        raise HTTPException(status_code=400, detail="Not initialized")

    new_genes_list = np.array(data.genes)
    migrants_integrated = state.integrate_migrants(new_genes_list)
    
    return {"status": "migrants_integrated", "count": migrants_integrated}

@app.get("/status")
def status():
    if not state.initialized or state.algorithm.opt is None:
        return {"sharpe": 0.0}
    return {"sharpe": float(state.algorithm.opt[0].get("sharpe"))}