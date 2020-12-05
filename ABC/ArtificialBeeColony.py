import numpy as np
import math
import time

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

class ArtificialBeeColony(object):

    def __init__(self, func, bounds, popSize=None, workerOnlookerSplit=0.5, limit=10, numScouts=1, crit="min", optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - workerOnlookerSplit: proportion between workers and onlookers. Greater values, more workers.
        - limit: for how many generations does a bee keep trying to improve solutions inside a region before turning into a scout.
        - scouts: how many scout bees are generated after the exhaustion of a "food source" at each generation.
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

        self.workerOnlookerSplit = workerOnlookerSplit
        self.limit = limit
        self.numScouts = numScouts

        # Control attributes
        self.workers = []
        self.patiences = []
        self.numWorkers = self.popSize * self.workerOnlookerSplit
        self.numOnlookers = self.popSize * (1 - self.workerOnlookerSplit)
        self.fVals = np.zeros(self.numWorkers)
        self.bestBeeIndexes = []
        self.worstBeeIndexes = []
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.results = None

        self.initializePopulation()


    def initializePopulation(self):

        self.workers = []

        # Population initialization as random (uniform)
        for i in range(self.numWorkers):

            self.workers.append( self.randomIndividual() )

        self.workers = np.array(self.workers) # result: matrix. Lines are individuals; columns are dimensions
        self.patiences = np.zeros(self.numWorkers)
        self.calculateFVals()

    def randomIndividual(self):

        return np.random.uniform(self.bounds[0], self.bounds[1])

    def calculateFVals(self):
        """Calculates all bees' objective function values. Also finds the worst and best bees' indexes."""

        # Fix: there is an index in common at the worst and best lists

        fVals = []
        bestFVal = np.inf if self.crit == "min" else -np.inf
        worstFVal = -np.inf if self.crit == "min" else np.inf
        self.bestBeeIndexes = []
        self.worstBeeIndexes = []

        for i in range(self.numWorkers):

            fVal = self.func(self.workers[i])
            self.FES += 1
            if self.FES == self.maxFES: raise MaxFESReached

            fVals.append(fVal)

            if(self.crit == "min"):

                if (fVal < bestFVal):
                    bestFVal = fVal
                    self.bestBeeIndexes = [ i ]

                elif (fVal == bestFVal):
                    self.bestBeeIndexes.append(i)

                if (fVal > worstFVal):
                    worstFVal = fVal
                    self.worstBeeIndexes = [ i ]

                elif (fVal == worstFVal):
                    self.worstBeeIndexes.append(i)

            else:

                if (fVal > bestFVal):
                    bestFVal = fVal
                    self.bestBeeIndexes = [ i ]

                elif (fVal == bestFVal):
                    self.bestBeeIndexes.append(i)

                if (fVal < worstFVal):
                    worstFVal = fVal
                    self.worstBeeIndexes = [ i ]

                elif (fVal == worstFVal):
                    self.worstBeeIndexes.append(i)

        self.fVals = fVals

    def makeScout(self, index):

        self.workers[index] = self.randomIndividual()
        self.patiences[index] = 0

    def improveSolution(self, index):

        candidate = self.workers[index]

        for i in range(self.dimensions):

            while True:

                bee2Index = np.random.randint(0, self.numWorkers)
                if bee2Index != index: break

            candidate[i] += np.random.uniform(-1, 1) * ( candidate[i] - self.workers[bee2Index][i] )

        candidate = self.checkNCorrectBounds(candidate)
        candidateFVal = self.calculateFVal(candidate)

        if self.crit == "min":

            if candidateFVal < self.fVals[index]:

                self.workers[index] = candidate
                self.patiences[index] = 0

            else:
                self.patiences[index] += 1

        else:

            if candidateFVal > self.fVals[index]:

                self.workers[index] = candidate
                self.patiences[index] = 0

            else:
                self.patiences[index] += 1


    def calculateFVal(self, bee):

        fVal = self.func(bee)
        self.FES += 1
        if self.FES == self.maxFES: raise MaxFESReached

        return fVal

    def createScouts(self):
        """Checks the patience values for every worker bee. Creates self.scouts scout bees, which do a random search.
        The scouts are created in the order of the workers whose patience has surpassed thelimit."""

        scoutCounter = 0

        while scoutCounter < self.numScouts:

            for i in range(self.numWorkers):

                if(self.patiences[i] > self.limit):

                self.makeScout(i)
                scoutCounter += 1

    def doOnlookers(self):
        """Operate with the onlooker bees. They randomly choose a position to improve, with probabilities proportional to the fitness value."""

        


    def checkNCorrectBounds(self, bee):
        """Bound checking and correcting function for the decision variables. If bounds are trespassed,
        the bee is truncated."""

        newBee = bee[:]

        for i in range( len(newBee) ):

            if(newBee[i] < self.bounds[0][i]):
                newBee[i] = self.bounds[0][i]
                # newBee[i] = np.random.uniform(self.bounds[0][1], self.bounds[0][1])
                # newBee[i] = 0

            if(newBee[i] > self.bounds[1][i]):
                newBee[i] = self.bounds[1][i]
                # newBee[i] = np.random.uniform(self.bounds[0][1], self.bounds[0][1])
                # newBee[i] = 0

        return newBee
