import numpy as np
import math
import time
from scipy.spatial.distance import pdist, squareform

class SocialSpiderAlgorithm(object):

    def __init__(self, func, bounds, popSize=None, crit="min", popSize=30, vibrationConstant=-700, optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - vibrationConstant: a constant used for calculating vibrations. Has to be lower than all possible values.
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


        # Control attributes
        self.spiders = np.zeros( shape=(self.popSize, self.dimensions) )
        self.fVals = np.zeros(self.popSize)
        self.pastTargets = np.zeros( shape=(self.popSize, self.dimensions) ) # last iteration's targers per spider
        self.iterationsSinceLastChange = np.zeros(self.popSize)
        self.pastMovements = np.zeros( shape=(self.popSize, self.dimensions) )
        self.pastMasks = [ [False for i in range(self.dimensions)] for i in range(self.popSize) ]
        self.selfVibrations = np.zeros(self.popSize)
        self.distances = np.zeros( shape=(self.popSize, self.popSize) )
        self.stdMean = 0
        self.bestSolutionIndexes = []
        self.worstSolutionIndexes = []
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.results = None

        # Archive initialization as random (uniform)
        for i in range(self.archiveSize):

            self.archive.append( self.randomSolution() )

        self.archive = np.array(self.archive) # result: matrix. Lines are individuals; columns are dimensions
        self.calculateArchiveFVals()
        self.rankArchive()
        self.calculateArchiveWeights()

    def randomSolution(self):
        return np.random.uniform(self.bounds[0], self.bounds[1])

    def vibration(self, index1, index2):

        if index1 == index2:

            if(self.crit == "min"):
                return np.log10( ( 1 / (self.fVals[index1] - self.vibrationConstant) ) + 1 )

            else:
                return np.log10(self.fVals[index1])

        else:

            return self.selfVibrations[index1] * np.exp( -(self.distances[index1][index2]) / (sigma? * ra?) )

    def calcStdMean(self):
        """Calculates the mean of the standard deviations at every dimension."""
        return base_distance = np.mean(np.std(self.spiders, 0))

    def calcDistances(self):
        """Calculates the Manhattan distances between all pairs of spiders. Saves it to self.distances."""
        self.distances = squareform(pdist(self.spiders, 'cityblock'))




if __name__ == '__main__':
    pass
