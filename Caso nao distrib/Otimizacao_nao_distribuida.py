import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import time


from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.repair import Repair
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.remote import Remote
from pymoo.core.termination import Termination
from pathlib import Path



#file = Remote.get_instance().load("examples", "portfolio_allocation.csv", to=None)
#df = pd.read_csv(file, parse_dates=True, index_col="date")




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
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        
        if exp_risk == 0:
            sharpe = 0
        else:
            sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = -sharpe
        out["risk_return"] = [exp_risk, exp_return]
        out["sharpe"] = sharpe


class SharpeStagnation(Termination):
    def __init__(self, n_last=30, tol=1e-3, max_gen=1000):
        """
        Critério híbrido: Para se estagnar OU se atingir o máximo de gerações.
        Isso evita o uso do operador '|' que causou o erro.
        """
        super().__init__()
        self.n_last = n_last
        self.tol = tol
        self.max_gen = max_gen
        self.history = []

    def _update(self, algorithm):
        if algorithm.n_gen >= self.max_gen:
            return 1.0

        if algorithm.opt is not None and len(algorithm.opt) > 0:
            current_fitness = algorithm.opt[0].F[0]
            self.history.append(current_fitness)

            if len(self.history) <= self.n_last:
                return 0.0

            past_fitness = self.history[-self.n_last]
            delta = abs(current_fitness - past_fitness)

            if delta < self.tol:
                print(f"\n[Critério de Parada] Estagnação detectada! Variação < {self.tol}")
                return 1.0
        
        return 0.0

# carrega portfolio_allocation.csv que está na mesma pasta do script (ou no cwd se __file__ não existir)
base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
csv_path = base / "portfolio_allocation.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

try:
    df = pd.read_csv(csv_path, parse_dates=True, index_col="date")
except Exception:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.set_index("date", inplace=True)



returns = df.pct_change().dropna(how="all")
mu = (1 + returns).prod() ** (252 / returns.count()) - 1
cov = returns.cov() * 252
mu_np, cov_np = mu.to_numpy(), cov.to_numpy()
labels = df.columns

start_time = time.time()
problem = PortfolioProblemGA(mu_np, cov_np)

algorithm = GA(
    pop_size=100,
    crossover=SBX(prob=0.9, eta=15, repair=PortfolioRepair()),
    mutation=PM(eta=20, repair=PortfolioRepair()),
    eliminate_duplicates=True
)

termination = SharpeStagnation(n_last=30, tol=1e-3, max_gen=1000)

print("Iniciando otimização...")
print("-" * 50)


res = minimize(
    problem,
    algorithm,
    termination=termination,
    seed=1,
    verbose=True
)

end_time = time.time()

execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = execution_time % 60

print("-" * 50)
print(f"Otimização concluída em: {minutes}m {seconds:.2f}s")
print(f"Gerações executadas: {res.algorithm.n_gen}")


if res.opt is not None:
    opt = res.opt[0]
    X_opt = opt.get("X")
    sharpe_opt = opt.get("sharpe")
    risk_ret_opt = opt.get("risk_return")

    pop = res.algorithm.pop
    risk_return_pop = pop.get("risk_return")
    risks = [ind[0] for ind in risk_return_pop]
    returns_plot = [ind[1] for ind in risk_return_pop]

    #plt.figure(figsize=(10, 6))
    #plt.scatter(risks, returns_plot, facecolor="none", edgecolors="blue", alpha=0.5, label="População Final")
    #plt.scatter(cov_np.diagonal() ** 0.5, mu_np, facecolor="none", edgecolors="black", s=30, label="Ativos Individuais")
    #plt.scatter(risk_ret_opt[0], risk_ret_opt[1], marker="*", s=200, color="red", label=f"Max Sharpe ({sharpe_opt:.4f})")
    #plt.title(f"Otimização de Portfólio (Tempo: {seconds:.2f}s)")
    #plt.xlabel("Volatilidade (Risco)")
    #plt.ylabel("Retorno Esperado")
    #plt.legend()
    #plt.grid(True, alpha=0.3)
    #plt.show()

    allocation = {name: w for name, w in zip(labels, X_opt) if w > 0}
    sorted_allocation = sorted(allocation.items(), key=operator.itemgetter(1), reverse=True)

    print("\n=== Melhor Alocação Encontrada ===")
    print(f"Sharpe Ratio: {sharpe_opt:.4f}")
    print(f"Retorno Esperado: {risk_ret_opt[1]:.2%}")
    print(f"Volatilidade (Risco): {risk_ret_opt[0]:.2%}")
    print("-" * 30)
    for name, w in sorted_allocation:
        print(f"{name:<10} | {w:.2%}")
else:
    print("Nenhuma solução ótima encontrada.")