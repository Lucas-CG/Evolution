import numpy as np
import math
import time
if(__name__ == '__main__'): from utilities.RouletteWheel import RouletteWheel
else: from .utilities.RouletteWheel import RouletteWheel

class MaxFESReached(Exception):
    """Exception used to interrupt the ACO operation when the maximum number of fitness evaluations is reached."""
    pass

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
        self.ants = np.zeros( shape=(self.numAnts, self.dimensions) )
        self.antFVals = np.zeros(self.numAnts)
        self.archive = [] # the archive also stores the means for each gaussian distribution - solution's value for the variable in question
        self.archiveFVals = np.zeros(self.archiveSize)
        self.archiveWeights = np.zeros(self.archiveSize)
        self.archiveStDevs = np.zeros( shape=(self.archiveSize, self.dimensions) ) - 1 # -1 indicates "not calculated"
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

    def calculateArchiveFVals(self):
        """Calculates all archive solutions' objective function values. Also finds the worst and best solutions' indexes."""

        fVals = []

        for i in range(self.archiveSize):

            fVal = self.func(self.archive[i])
            self.FES += 1
            if self.FES == self.maxFES: raise MaxFESReached

            fVals.append(fVal)

        self.archiveFVals = np.array(fVals)

    def getFirst(self, ind):
        return ind[0]

    def rankArchive(self): # after individual updates
        """Orders the archive's solutions in decrescent order of fitness (minimization: crescent order of obj. func.'s value).
         Also updates the best and worst solutions of the population."""

        bestFVal = np.inf if self.crit == "min" else -np.inf
        worstFVal = -np.inf if self.crit == "min" else np.inf
        self.bestSolutionIndexes = []
        self.worstSolutionIndexes = []

        valsAndOrders = [ (self.archiveFVals[i], i) for i in range(self.archiveSize) ]

        if(self.crit == "min"): valsAndOrders.sort(key=self.getFirst) # crescent order (min)
        else: valsAndOrders.sort(key=self.getFirst, reverse=True) # decrescent order (max)
        # Will return a sorted list. First item of each element is the objective
        # function's value; the second one is the original index.

        bestFVal = valsAndOrders[0][0]
        worstFVal = valsAndOrders[-1][0]

        for i in range(self.archiveSize):

            if (valsAndOrders[i][0] == bestFVal):
                self.bestSolutionIndexes.append(i)

            else: break

        for i in reversed( range(self.archiveSize) ):

            if (valsAndOrders[i][0] == worstFVal):
                self.worstSolutionIndexes.append(i)

            else: break

        self.archiveFVals = [ element[0] for element in valsAndOrders ]

        self.archive = np.array( [ self.archive[element[1]] for element in valsAndOrders ] )

    def calculateWeight(self, rank):
        """Calculates the weight of a solution of the archive. Supposes that the archive is already sorted and ranked."""

        return ( 1 / ( self.searchLocality * self.archiveSize * np.sqrt(2 * np.pi) ) ) * \
                np.exp( -( (rank - 1) ** 2 ) / 2 * self.searchLocality**2 * self.archiveSize**2 )

    def calculateArchiveWeights(self):
        """Calculates the weights of the entire archive. Required for moving the ants."""

        for i in range(self.archiveSize): self.archiveWeights[i] = self.calculateWeight(i + 1)
        # i + 1 because ranks start from 1

    def calculateStDev(self, index, dimension):
        """Calculates the standard deviation of the normal distribution of an archive solution
        for a specified dimension."""

        total = 0

        for e in range(self.archiveSize):

            if e == index: continue
            else: total += abs(self.archive[e][dimension] - self.archive[index][dimension])

        self.archiveStDevs[index][dimension] = (self.convergenceSpeed * total) / (self.archiveSize - 1)

    def moveAnts(self):
        """Moves the ants, considering the solutions in the archive. Supposes that the archive
        is ranked by objective function values and that the weights are already calculated."""

        roulette = RouletteWheel(self.archiveWeights)

        for i in range(self.numAnts):

            chosenSolution = roulette.draw()

            for j in range(self.dimensions):

                # Calculates the st. dev. if it is not already calculated
                if(self.archiveStDevs[chosenSolution][j] == -1.0): self.calculateStDev(chosenSolution, j)

                # Samples this variable's value from a normal distribution.
                # Mean: the chosen archive's solution value.
                # Standard deviation: the calculated standard deviation for this archive solution and variable.
                self.ants[i][j] = np.random.normal(loc=self.archive[chosenSolution][j], scale=self.archiveStDevs[chosenSolution][j])

            self.ants[i] = self.checkNCorrectBounds(self.ants[i])

    def checkNCorrectBounds(self, solution):
        """Bound checking and correcting function for the decision variables. If bounds are trespassed,
        the solution is truncated."""

        newSolution = np.copy(solution)

        for i in range( len(newSolution) ):

            if(newSolution[i] < self.bounds[0][i]):
                newSolution[i] = self.bounds[0][i]
                # newSolution[i] = np.random.uniform(self.bounds[0][1], self.bounds[0][1])
                # newSolution[i] = 0

            if(newSolution[i] > self.bounds[1][i]):
                newSolution[i] = self.bounds[1][i]
                # newSolution[i] = np.random.uniform(self.bounds[0][1], self.bounds[0][1])
                # newSolution[i] = 0

        return newSolution

    def evaluateAnts(self):
        """Calculates the ants' objective function values."""

        fVals = []

        for i in range(self.numAnts):

            fVal = self.func(self.ants[i])
            self.FES += 1
            if self.FES == self.maxFES: raise MaxFESReached

            fVals.append(fVal)

        self.antFVals = np.array(fVals)

    def rankAnts(self):
        """Orders the ants' found solutions in decrescent order of fitness (minimization: crescent order of obj. func.'s value)."""

        valsAndOrders = [ (self.antFVals[i], i) for i in range(self.numAnts) ]

        if(self.crit == "min"): valsAndOrders.sort(key=self.getFirst) # crescent order (min)
        else: valsAndOrders.sort(key=self.getFirst, reverse=True) # decrescent order (max)
        # Will return a sorted list. First item of each element is the objective
        # function's value; the second one is the original index.

        self.antFVals = [ element[0] for element in valsAndOrders ]

        self.ants = np.array( [ self.ants[element[1]] for element in valsAndOrders ] )

    def replaceArchiveByAnts(self):
        """Replaces the worst archive solutions by the current iteration's ants, if they have found
        better solutions."""

        # The procedure keeps two counters, i and j, which respectively traverse the archive and the
        # ant list in reverse order. This way, we iterate from the worst to the best solutions, since
        # both lists are ranked, and do not lose solutions (which would happen if the lists were traversed in
        # opposite directions).
        # When we find an ant with a better solution than the current archive solution (given by i), we replace it by the ant.
        # Replacing or not, the counter j moves, changing the ant. If there was a replacement, we go to the next ant
        # to avoid repetitions. If there wasn't, we try a better ant candidate to replace the current archive solution.
        # It is important to initialize j BEFORE the for loop (that iterates through the archive). Otherwise,
        # j would be reinitialize and we would reevaluate ants, which might lead to the archive being full of
        # repeated solutions. Also, in this case, there would be more replacements than ants.

        j = self.numAnts - 1 # ant counter(reverse order)

        for i in reversed( range(self.archiveSize) ):

            while j >= 0:

                if(self.crit == "min" and self.antFVals[j] < self.archiveFVals[i]):

                    self.archive[i] = self.ants[j]
                    self.archiveFVals[i] = self.antFVals[j]
                    j -= 1
                    break

                elif(self.crit == "max" and self.antFVals[j] > self.archiveFVals[i]):

                    self.archive[i] = self.ants[j]
                    self.archiveFVals[i] = self.antFVals[j]
                    j -= 1
                    break

                j -= 1

            if (j < 0): break

    def resetStDevs(self):
        """Resets the archive's standard deviations. Required at the end of every iteration."""

        self.archiveStDevs = np.zeros( shape=(self.archiveSize, self.dimensions) ) - 1 # -1 indicates "not calculated"

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

        Execute only after evaluating and ranking the archive's solutions!"""

        avg = sum(self.archiveFVals)/self.archiveSize
        bestVal = self.archiveFVals[self.bestSolutionIndexes[0]]
        bestPoints = [ self.archive[i] for i in self.bestSolutionIndexes ]
        worstVal = self.archiveFVals[self.worstSolutionIndexes[0]]
        worstPoints = [ self.archive[i] for i in self.worstSolutionIndexes ]

        error = abs(bestVal - self.optimum)

        return {"avg": avg, "bestVal": bestVal, "bestPoints": bestPoints, "worstVal": worstVal, "worstPoints": worstPoints, "error": error}

    def execute(self):

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

            while ( abs(self.archiveFVals[self.bestSolutionIndexes[0]] - self.optimum) > self.tol ):

                self.moveAnts()

                try:
                    self.evaluateAnts()

                except MaxFESReached:
                    break

                self.rankAnts()
                self.replaceArchiveByAnts()

                # Resetting archive's st. devs., reranking it and reweighting it
                self.resetStDevs()
                self.rankArchive()
                self.calculateArchiveWeights()

                # Getting metrics based on present archive, and FES and generation counters.
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

                # print(metrics["error"])

        except KeyboardInterrupt:
            return

if __name__ == '__main__':

    # Test of the ACO's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005
    # np.seterr("raise") # any calculation error immediately stops the execution
    # dims = 10
    #
    # bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ] # 10-dimensional sphere (optimum: 0)
    #
    # start = time.time()
    #
    # # Initialization
    # ACO = AntColonyOptimization(cec2005.F2(dims), bounds, optimum=-450) # F5: -310 / others: -450
    # ACO.execute()
    # results = ACO.results
    #
    # print("ACO: for criterion = " + ACO.crit + ", reached optimum of " + str(results["bestFits"][-1]) +
    # " (error of " + str(results["errors"][-1]) + ") (points " + str(results["bestPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    # " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )
    #
    # end = time.time()
    # print("time:" + str(end - start))

    import sys
    sys.path.append("../../cec2014/python") # Fedora
    # sys.path.append("/mnt/c/Users/Lucas/Documents/git/cec2014/python") # Windows
    import cec2014

    def func(arr):
        return cec2014.cec14(arr, 1)

    dims = 10

    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    ACO = AntColonyOptimization(func, bounds, numAnts=2, archiveSize=30, optimum=100) # F5: -310 / others: -450
    ACO.execute()
    results = ACO.results

    print("ACO: for criterion = " + ACO.crit + ", reached optimum of " + str(results["bestFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["bestPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
