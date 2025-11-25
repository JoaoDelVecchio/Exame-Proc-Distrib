import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from portifolio_problem import *


class IslandState:
    def __init__(self):
        self.algorithm = None
        self.problem = None
        self.initialized = False

    def initialize(self, mu_np, cov_np):
        self.problem = PortfolioProblemGA(mu_np, cov_np)

        self.algorithm = GA(
            pop_size=100,
            crossover=SBX(prob=0.9, eta=15, repair=PortfolioRepair()),
            mutation=PM(eta=20, repair=PortfolioRepair()),
            eliminate_duplicates=True
        )

        self.algorithm.setup(self.problem)
        self.initialized = True


    def integrate_migrants(self, new_genes_list):
        if new_genes_list.ndim == 1:
            new_genes_list = new_genes_list.reshape(1, -1)

        migrant_pop = Population.new("X", new_genes_list)

        Evaluator().eval(self.problem, migrant_pop)

        main_pop = self.algorithm.pop
        F_main = main_pop.get("F").flatten()
        worst_indices = np.argsort(F_main)[-len(migrant_pop):]

        for i, idx in enumerate(worst_indices):
            main_pop[idx] = migrant_pop[i]

        return len(migrant_pop)