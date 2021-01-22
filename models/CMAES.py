import cma # https://github.com/CMA-ES/pycma
import numpy as np

class CMAES(object):
    """Wrapper for the CMA-ES/pycma implementation for minimization."""

    def __init__(self, func, bounds, popSize=None, optimum=-450, maxFES=None, tol=1e-08):
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
        self.optimum = optimum
        self.tol = tol
        self.crit = "min"

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * self.dimensions

        # Control attributes
        self.genCount = 0
        self.bestSoFar = 0
        self.results = None
        self.X = None
        self.Y = None

        if( len(self.bounds[0]) != len(self.bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        self.pop = []

        startPoint = [np.random.uniform(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dimensions)]
        # self.es = cma.CMAEvolutionStrategy(startPoint, 60.0, {'bounds': [-100, +100],"maxfevals": self.maxFES, "maxiter": np.inf, "verbose": -1} )
        if (self.popSize == None): self.es = cma.CMAEvolutionStrategy(startPoint, 60.0, {'bounds': [-100, +100],"maxfevals": self.maxFES, "maxiter": np.inf, "tolstagnation": np.inf, "tolfun": -np.inf, "tolflatfitness": np.inf, "tolfunhist": -np.inf, "verbose": -1} )
        else: self.es = cma.CMAEvolutionStrategy(startPoint, 60.0, {'bounds': [-100, +100],"maxfevals": self.maxFES, "maxiter": np.inf, "tolstagnation": np.inf, "tolfun": -np.inf, "tolflatfitness": np.inf, "tolfunhist": -np.inf, "verbose": -1, "popsize": self.popSize} )
        # most tolerances are disabled - for analysis -, except 'noeffectaxis'


    def execute(self):

        generations = []
        FESCount = []
        errors = []
        maxFits = []
        minFits = []
        avgFits = []

        while not self.es.stop() and abs(self.es.result.fbest - self.optimum) > self.tol:
            self.X = self.es.ask()
            self.Y = [self.func(np.array(x)) for x in self.X]
            self.es.tell(self.X, self.Y)

            metrics = self.getFitnessMetrics()
            generations.append(self.es.result.iterations)
            FESCount.append(self.es.result.evaluations)
            errors.append(metrics["error"])
            maxFits.append(metrics["top"])
            minFits.append(metrics["bottom"])
            avgFits.append(metrics["avg"])

            # print(self.es.result.iterations, self.es.result.evaluations, metrics["error"])

            self.results = {"generations": generations,
                "FESCounts": FESCount,
                "errors": errors,
                "maxFits": maxFits,
                "minFits": minFits,
                "avgFits": avgFits}


    def getFitnessMetrics(self):

        """Finds the mean, greater and lower fitness values for the population,
        as well as the points with the greater and lower ones.
        Returns a dict, whose keys are:
        "avg" to average value
        "top" to top value
        "bottom" to bottom value

        Execute after evaluating fitness values for the entire population!"""

        top = max(self.Y)
        bottom = self.es.result.fbest
        avg = np.mean(np.array(self.Y))
        self.bestSoFar = self.es.result.fbest
        error = abs(self.bestSoFar - self.optimum)

        return {"avg": avg, "top": top, "bottom": bottom, "error": error}


if __name__ == '__main__':

    # Test of the CMAES's performance over CEC2005's F3

    import time
    from optproblems import cec2005

    import sys
    sys.path.append("../../cec2014/python") # Fedora
    # sys.path.append("/mnt/c/Users/Lucas/Documents/git/cec2014/python") # Windows
    import cec2014

    def func(arr):
        return cec2014.cec14(arr, 6)

    dims = 30
    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    # es = CMAES(cec2005.F3(dims), bounds, optimum=-450)
    es = CMAES(func, bounds, popSize=300, optimum=600)
    es.execute()
    results = es.results

    print("CMA-ES: for criterion = " + es.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
