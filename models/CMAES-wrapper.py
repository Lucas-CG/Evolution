import cma
import numpy as np

class CMAES(object):
    """Wrapper for the CMA-ES/pycma implementation."""

    def __init__(self, func, bounds, popSize=100, crit="min", eliteSize=0, matingPoolSize=100, numChildren=100, adaptiveEpsilon=0.1, optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function name (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - crit: criterion ("min" or "max")
        - eliteSize: positive integer; defines whether elitism is enabled or not
        - matingPoolSize: indicate the size of the mating pool
        - optimum: known optimum value for the objective function. Default is 0.
        - maxFES: maximum number of fitness evaluations.
        If set to None, will be calculated as 10000 * [number of dimensions] = 10000 * len(bounds)"""

        # Attribute initialization

        # From arguments
        self.func = func
        self.bounds = bounds
        self.dimensions = len(self.bounds[0])
        self.popSize = popSize
        self.crit = crit
        self.eliteSize = eliteSize
        self.optimum = optimum
        self.tol = tol
        self.matingPoolSize = matingPoolSize
        self.numChildren = numChildren
        self.adaptiveEpsilon = adaptiveEpsilon

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * len(bounds[0]) # 10000 x [dimensions]

        # Control attributes
        self.pop = None
        self.matingPool = None # used for parent selection
        self.children = None
        self.elite = None # used in elitism
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.bestSoFar = 0
        self.mutation = None
        self.mutationParams = None
        self.parentSelection = None
        self.parentSelectionParams = None
        self.newPopSelection = None
        self.newPopSelectionParams = None
        self.results = None

        if( len(self.bounds[0]) != len(self.bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        self.pop = []

        # Population initialization as random (uniform)
        for i in range(self.popSize):
            ind = []
            genes = np.random.uniform(self.bounds[0], self.bounds[1]).tolist()
            sigmas = np.random.uniform(0.1, 1, self.dimensions).tolist()
            # tolist(): convert to python list
            genes.extend(sigmas)
            self.pop.append([genes, 0])

        es = cma.CMAEvolutionStrategy(self.dimensions * [0.5], 0.2, {'bounds': [-100, +100]},
        "maxfevals": 10000 * self.dimensions, "maxiter": np.inf, "tolstagnation": np.inf)

        self.calculateFitnessPop()


while not es.stop():
    X = es.ask()
    es.tell(X, [self.func(np.array(s)) for s in solutions])
    es.disp()
