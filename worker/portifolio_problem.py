import numpy as np
from pymoo.core.repair import Repair
from pymoo.core.problem import ElementwiseProblem

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