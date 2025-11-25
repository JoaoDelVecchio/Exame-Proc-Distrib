import numpy as np
import logging
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from portifolio_problem import *
from constants import *

class IslandState:
    def __init__(self):
        self.algorithm = None
        self.problem = None
        self.initialized = False

    def initialize(self, mu_np, cov_np):
        self.problem = PortfolioProblemGA(mu_np, cov_np)

        self.algorithm = GA(
            pop_size=POPULATION_SIZE,
            crossover=SBX(prob=CROSSOVER, eta=CROSSOVER_ETA, repair=PortfolioRepair()),
            mutation=PM(eta=MUTATION_ETA, repair=PortfolioRepair()),
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

        try:
            old_sharpes = main_pop[worst_indices].get("sharpe")
        except Exception:
            old_sharpes = None
        try:
            new_sharpes = migrant_pop.get("sharpe")
        except Exception:
            new_sharpes = None

        logging.info(f"Integrating {len(migrant_pop)} migrants replacing indices {worst_indices.tolist()} | old_sharpes={old_sharpes} -> new_sharpes={new_sharpes}")

        for i, idx in enumerate(worst_indices):
            logging.debug(f"Replacing index {idx} (old_sharpe={None if old_sharpes is None else float(old_sharpes[i])}) with migrant {i} (new_sharpe={None if new_sharpes is None else float(new_sharpes[i])})")
            main_pop[idx] = migrant_pop[i]

        return len(migrant_pop)