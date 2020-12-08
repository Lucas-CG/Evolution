import numpy as np
import math
import time
if(__name__ == '__main__'): from utilities.RouletteWheel import RouletteWheel
else: from .utilities.RouletteWheel import RouletteWheel

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

class ArtificialBeeColony(object):

    def __init__(self, func, bounds, popSize=None, workerOnlookerSplit=0.5, limit=None, numScouts=1, crit="min", optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - workerOnlookerSplit: proportion between workers and onlookers. Greater values, more workers.
        - limit: for how many generations does a bee keep trying to improve solutions inside a region before turning into a scout.
        - numScouts: how many scout bees are generated after the exhaustion of a "food source" at each generation.
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
        self.numScouts = numScouts

        # Control attributes
        self.workers = []
        self.patiences = []
        self.numWorkers = int(np.floor(self.popSize * self.workerOnlookerSplit))
        self.numOnlookers = self.popSize - self.numWorkers
        self.limit = limit if limit else 0.6 * self.numWorkers * self.dimensions
        self.fVals = np.zeros(self.numWorkers)
        self.bestBeeIndexes = []
        self.worstBeeIndexes = []
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.results = None

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

    def updateBestWorstBees(self): # after individual updates
        """Updates the best and worst bees of the population."""

        bestFVal = np.inf if self.crit == "min" else -np.inf
        worstFVal = -np.inf if self.crit == "min" else np.inf
        self.bestBeeIndexes = []
        self.worstBeeIndexes = []

        for i in range(self.numWorkers):

            if(self.crit == "min"):

                if (self.fVals[i] < bestFVal):
                    bestFVal = self.fVals[i]
                    self.bestBeeIndexes = [ i ]

                elif (self.fVals[i] == bestFVal):
                    self.bestBeeIndexes.append(i)

                if (self.fVals[i] > worstFVal):
                    worstFVal = self.fVals[i]
                    self.worstBeeIndexes = [ i ]

                elif (self.fVals[i] == worstFVal):
                    self.worstBeeIndexes.append(i)

            else:

                if (self.fVals[i] > bestFVal):
                    bestFVal = self.fVals[i]
                    self.bestBeeIndexes = [ i ]

                elif (self.fVals[i] == bestFVal):
                    self.bestBeeIndexes.append(i)

                if (self.fVals[i] < worstFVal):
                    worstFVal = self.fVals[i]
                    self.worstBeeIndexes = [ i ]

                elif (self.fVals[i] == worstFVal):
                    self.worstBeeIndexes.append(i)

    def calculateFVal(self, bee):

        fVal = self.func(bee)
        self.FES += 1
        if self.FES == self.maxFES: raise MaxFESReached

        return fVal

    def improveSolution(self, index):
        """Improves a solution by modifying one random coordinate with a perturbation."""

        candidate = np.copy(self.workers[index])

        i = np.random.randint(0, self.dimensions)

        bee2Index = index

        while bee2Index == index: bee2Index = np.random.randint(0, self.numWorkers)

        candidate[i] += np.random.uniform(-1, 1) * ( candidate[i] - self.workers[bee2Index][i] )

        candidate = self.checkNCorrectBounds(candidate)
        candidateFVal = self.calculateFVal(candidate)

        if self.crit == "min":

            if candidateFVal < self.fVals[index]:

                self.workers[index] = np.copy(candidate)
                self.fVals[index] = candidateFVal
                self.patiences[index] = 0

            else:
                self.patiences[index] += 1

        else:

            if candidateFVal > self.fVals[index]:

                self.workers[index] = np.copy(candidate)
                self.fVals[index] = candidateFVal
                self.patiences[index] = 0

            else:
                self.patiences[index] += 1


    def doWorkers(self):
        """Worker bees operate, by trying to improve their solution ("food source")."""

        for i in range(self.numWorkers):

            self.improveSolution(i)

    def makeScout(self, index):
        """Create a scout bee at an specified index, i.e., resets its position to a random location and resets its patience to 0."""

        self.workers[index] = self.randomIndividual()
        self.fVals[index] = self.calculateFVal(self.workers[index])
        self.patiences[index] = 0

    def createScouts(self):
        """Checks the patience values for every worker bee. Creates self.scouts scout bees, which do a random search.
        The scouts are created in the order of the workers whose patience has surpassed the limit (self.limit)."""

        scoutCounter = 0
        i = 0

        while scoutCounter < self.numScouts and i < self.numWorkers:

            pats = self.patiences.tolist()
            j = pats.index(max(pats)) # prioritizes the bee with the most iterations

            if(self.patiences[j] > self.limit):

                self.makeScout(j)
                scoutCounter += 1

            i += 1

    def doOnlookers(self, method="roulette"):
        """Operate with the onlooker bees. They randomly choose a position to improve, with probabilities proportional to the fitness value."""

        weights = []

        for i in range(self.numWorkers):

            if self.crit == "min":

                weight = 1/(1 + self.fVals[i]) if self.fVals[i] >= 0 else 1 + abs(self.fVals[i]) # Akay, Karaboga, 2012
                weights.append(weight)

            else: weights.append(self.fVals[i])

        weights = np.array(weights)

        if(method == "roulette"):

            roulette = RouletteWheel(weights)

            for i in range(self.numOnlookers):

                self.improveSolution(roulette.draw()) # improves the bee that is selected by the roulette

    def checkNCorrectBounds(self, bee):
        """Bound checking and correcting function for the decision variables. If bounds are trespassed,
        the bee is truncated."""

        newBee = np.copy(bee)

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

    def getFitnessMetrics(self):

        """Finds the mean, greater and lower fitness values for the population,
        as well as the points with the greater and lower ones and the current error.
        Returns a dict, whose keys are:
        "avg" to average value
        "bestVal" to best value
        "bestPoints" to a list of points with the best value
        "worstVal" to worst value
        "worstPoints" to a list of points with the worst value
        "error" for the current error (difference between the fitness and the optimum)

        Execute only after using self.calculateFVals and self.improveSolution!"""

        avg = sum(self.fVals)/self.numWorkers
        bestVal = self.fVals[self.bestBeeIndexes[0]]
        bestPoints = [ self.workers[i] for i in self.bestBeeIndexes ]
        worstVal = self.fVals[self.worstBeeIndexes[0]]
        worstPoints = [ self.workers[i] for i in self.worstBeeIndexes ]

        error = abs(bestVal - self.optimum)

        return {"avg": avg, "bestVal": bestVal, "bestPoints": bestPoints, "worstVal": worstVal, "worstPoints": worstPoints, "error": error}

    def execute(self):

        self.initializePopulation()
        metrics = self.getFitnessMetrics() # post-initialization: generation 0

        # Arrays for collecting metrics

        generations = [ self.genCount ]
        FESCount = [ self.FES ]
        errors = [ metrics["error"] ]
        bestFits = [ metrics["bestVal"] ]
        bestPoints = [ metrics["bestPoints"] ]
        worstFits = [ metrics["worstVal"] ]
        worstPoints = [ metrics["worstPoints"] ]
        avgFits = [ metrics["avg"] ]

        try:

            while ( abs(self.fVals[self.bestBeeIndexes[0]] - self.optimum) > self.tol ):

                try:
                    self.doWorkers()

                except MaxFESReached:
                    break

                try:
                    self.doOnlookers()

                except MaxFESReached:
                    break

                try:
                    self.createScouts()

                except MaxFESReached:
                    break

                self.updateBestWorstBees()
                metrics = self.getFitnessMetrics()

                self.genCount += 1

                generations.append(self.genCount)
                FESCount.append(self.FES)
                errors.append(metrics["error"])
                bestFits.append(metrics["bestVal"])
                bestPoints.append(metrics["bestPoints"])
                worstFits.append(metrics["worstVal"])
                worstPoints.append(metrics["worstPoints"])
                avgFits.append(metrics["avg"])

                self.results = {"generations": generations,
                    "FESCounts": FESCount,
                    "errors": errors,
                    "bestFits": bestFits,
                    "bestPoints": bestPoints,
                    "worstFits": worstFits,
                    "worstPoints": worstPoints,
                    "avgFits": avgFits}

        except KeyboardInterrupt:
            return


if __name__ == '__main__':

    # Test of the ABC's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005
    # np.seterr("raise") # any calculation error immediately stops the execution
    dims = 10

    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    ABC = ArtificialBeeColony(cec2005.F1(dims), bounds, popSize=50, workerOnlookerSplit=0.5, limit=None, numScouts=1, optimum=-450) # F5: -310 / others: -450
    ABC.execute()
    results = ABC.results

    print("ABC: for criterion = " + ABC.crit + ", reached optimum of " + str(results["bestFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["bestPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
