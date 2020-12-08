import numpy as np
import math
import time

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

def getSecond(ind):
    return ind[1]

class AntColonyOptimization(object):

    def __init__(self, func, bounds, popSize=None, crit="min", numAnts=2, archiveSize=50, convergenceSpeed=0.85, searchLocality=1e-04, optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - numAnts: number of ants used in an iteration
        - archiveSize: amount of solutions stored in the archive ("feromones")
        - convergenceSpeed: a real-valued constant. The greater it is, the lower the convergence speed. It is used for calculating standard deviations.
        - searchLocality: a real-valued constant. The lower it is, the more the algorithm favors the best-ranked solutions in the archive.
        Used for calculating the weights of solutions in the archive.
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

        self.numAnts = numAnts
        if archiveSize < self.dimensions: raise ValueError("ACO: archive size is lower than the number of dimensions.")
        self.archiveSize = archiveSize
        self.convergenceSpeed = convergenceSpeed
        self.searchLocality = searchLocality

        # Control attributes
        self.ants = np.zeros(self.numAnts)
        self.archive = [] # the archive also stores the means for each gaussian distribution - solution's value for the variable in question
        self.archiveFVals = np.zeros(self.archiveSize)
        self.archiveWeights = np.zeros(self.archiveSize)
        self.archiveStDevs = np.array( [ -1 for i in range(self.archiveSize) ] ) # -1 indicates "not calculated"
        self.ranks = [i+1 for i in range self.archiveSize]
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

    def randomSolution(self):
        return np.random.uniform(self.bounds[0], self.bounds[1])

    def calculateArchiveFVals(self):
        """Calculates all archive solutions' objective function values. Also finds the worst and best solutions' indexes."""

        fVals = []
        bestFVal = np.inf if self.crit == "min" else -np.inf
        worstFVal = -np.inf if self.crit == "min" else np.inf
        self.bestSolutionIndexes = []
        self.worstSolutionIndexes = []

        for i in range(self.archiveSize):

            fVal = self.func(self.archive[i])
            self.FES += 1
            if self.FES == self.maxFES: raise MaxFESReached

            fVals.append(fVal)

            if(self.crit == "min"):

                if (fVal < bestFVal):
                    bestFVal = fVal
                    self.bestSolutionIndexes = [ i ]

                elif (fVal == bestFVal):
                    self.bestSolutionIndexes.append(i)

                if (fVal > worstFVal):
                    worstFVal = fVal
                    self.worstSolutionIndexes = [ i ]

                elif (fVal == worstFVal):
                    self.worstSolutionIndexes.append(i)

            else:

                if (fVal > bestFVal):
                    bestFVal = fVal
                    self.bestSolutionIndexes = [ i ]

                elif (fVal == bestFVal):
                    self.bestSolutionIndexes.append(i)

                if (fVal < worstFVal):
                    worstFVal = fVal
                    self.worstSolutionIndexes = [ i ]

                elif (fVal == worstFVal):
                    self.worstSolutionIndexes.append(i)

        self.archiveFVals = np.array(fVals)

    def updateBestWorstSolutions(self): # after individual updates
        """Updates the best and worst solutions of the population."""

        bestFVal = np.inf if self.crit == "min" else -np.inf
        worstFVal = -np.inf if self.crit == "min" else np.inf
        self.bestSolutionIndexes = []
        self.worstSolutionIndexes = []

        for i in range(self.archiveSize):

            if(self.crit == "min"):

                if (self.fVals[i] < bestFVal):
                    bestFVal = self.fVals[i]
                    self.bestSolutionIndexes = [ i ]

                elif (self.fVals[i] == bestFVal):
                    self.bestSolutionIndexes.append(i)

                if (self.fVals[i] > worstFVal):
                    worstFVal = self.fVals[i]
                    self.worstSolutionIndexes = [ i ]

                elif (self.fVals[i] == worstFVal):
                    self.worstSolutionIndexes.append(i)

            else:

                if (self.fVals[i] > bestFVal):
                    bestFVal = self.fVals[i]
                    self.bestSolutionIndexes = [ i ]

                elif (self.fVals[i] == bestFVal):
                    self.bestSolutionIndexes.append(i)

                if (self.fVals[i] < worstFVal):
                    worstFVal = self.fVals[i]
                    self.worstSolutionIndexes = [ i ]

                elif (self.fVals[i] == worstFVal):
                    self.worstSolutionIndexes.append(i)

    def calculateFVal(self, solution):

        fVal = self.func(solution)
        self.FES += 1
        if self.FES == self.maxFES: raise MaxFESReached

        return fVal

    def calculateWeight(self, rank):

        return ( 1 / ( self.searchLocality * self.archiveSize * np.sqrt(2 * np.pi) ) ) * \
                np.exp( -( (rank - 1) ** 2 ) / )

    def calculateStDev(self, index, dimension):

        self.convergenceSpeed
        total = 0

        for e in range(self.archiveSize):

            if e == index: continue
            else: total += abs(self.archive[e][dimension] - self.archive[index][dimension])

        return (convergenceSpeed * total) / (self.archiveSize - 1)
