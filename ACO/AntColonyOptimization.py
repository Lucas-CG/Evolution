import numpy as np
import math
import time

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass


class AntColonyOptimization(object):

    def __init__(self, func, bounds, popSize=None, crit="min", archive_size=10, optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - crit: criterion ("min" or "max")
        - optimum: known optimum value for the objective function. Default is -450, for CEC functions.
        - maxFES: maximum number of fitness evaluations.
        If set to None, will be calculated as 10000 * [number of dimensions] = 10000 * len(bounds)"""

        # Attribute initialization

        # From arguments

        if( len(bounds[0]) != len(bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        self.func = func
        self.bounds = bounds
        self.crit = crit
        self.optimum = optimum
        self.tol = tol
        self.dimensions = len(self.bounds[0])

        if(popSize): self.popSize = popSize
        else: self.popSize = 10 * self.dimensions

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * self.dimensions


        if archive_size < self.dimensions: raise ValueError("ACO: archive size is lower than the number of dimensions.")
        self.archive_size = archive_size

        # Control attributes
        self.ants = []
        self.fVals = np.zeros(self.numWorkers)
        self.bestAntIndexes = []
        self.worstAntIndexes = []
        self.archive = np.zeros(self.archive_size, self.dimensions)
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.results = None

    def initializePopulation(self):

        self.ants = []

        # Population initialization as random (uniform)
        for i in range(self.popSize):

            self.ants.append( self.randomIndividual() )

        self.ants = np.array(self.ants) # result: matrix. Lines are individuals; columns are dimensions
        self.calculateFVals()

    def randomIndividual(self):
        return np.random.uniform(self.bounds[0], self.bounds[1])
