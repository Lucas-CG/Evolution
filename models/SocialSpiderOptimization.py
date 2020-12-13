import numpy as np
import math
import time

#IMPORTANT CHANGE: USED LOG1P FOR DISTANCE (DISTANCES ARE TOO LARGE!)

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

class SocialSpiderOptimization(object):
    """Implements a real-valued Social Spider Optimization."""

    def __init__(self, func, bounds, popSize=None, PF=0.7, normalizeDistances=True, crit="min", optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - PF: probability of attraction (for females)
        - normalizeDistances: divide all distances by the diameter (maximum possible distance), to avoid np.exp underflows.
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
        self.PF = PF # probability of attraction (female movements)

        if(popSize): self.popSize = popSize
        else: self.popSize = 10 * self.dimensions

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * self.dimensions

        self.normalizeDistances = normalizeDistances

        # Control attributes
        self.spiders = []
        self.numFemales = int(np.floor( (0.9 - np.random.uniform(0, 0.25)) * self.popSize ))
        self.numMales = self.popSize - self.numFemales
        self.fVals = np.zeros(self.popSize)
        self.weights = np.zeros(self.popSize)
        self.bestSpiderIndexes = []
        self.worstSpiderIndexes = []
        self.isDominant = None # flags for identifying dominant males
        self.matingRadius = sum( np.array(self.bounds[1]) - np.array(self.bounds[0]) ) / (2 * self.dimensions)
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.results = None

        # Population initialization as random (uniform)
        for i in range(self.popSize):

            self.spiders.append( np.random.uniform(self.bounds[0], self.bounds[1]) )

        self.spiders = np.array(self.spiders) # result: matrix. Lines are individuals; columns are dimensions
        # first lines are females; last ones are males

        self.diameter = 0 # greatest distance for the interval

        if(self.normalizeDistances):
            extreme0 = np.array(self.bounds[0])
            extreme1 = np.array(self.bounds[1])
            self.diameter = np.linalg.norm( extreme0 - extreme1 )

    def calculateFVals(self):
        """Calculates all spiders' objective function values. Also finds the worst and best spiders' indexes."""

        fVals = []
        bestFVal = np.inf if self.crit == "min" else -np.inf
        worstFVal = -np.inf if self.crit == "min" else np.inf
        self.bestSpiderIndexes = []
        self.worstSpiderIndexes = []

        for i in range(self.popSize):

            fVal = self.func(self.spiders[i])
            self.FES += 1
            if self.FES == self.maxFES: raise MaxFESReached

            fVals.append(fVal)

            if(self.crit == "min"):

                if (fVal < bestFVal):
                    bestFVal = fVal
                    self.bestSpiderIndexes = [ i ]

                elif (fVal == bestFVal):
                    self.bestSpiderIndexes.append(i)

                if (fVal > worstFVal):
                    worstFVal = fVal
                    self.worstSpiderIndexes = [ i ]

                elif (fVal == worstFVal):
                    self.worstSpiderIndexes.append(i)

            else:

                if (fVal > bestFVal):
                    bestFVal = fVal
                    self.bestSpiderIndexes = [ i ]

                elif (fVal == bestFVal):
                    self.bestSpiderIndexes.append(i)

                if (fVal < worstFVal):
                    worstFVal = fVal
                    self.worstSpiderIndexes = [ i ]

                elif (fVal == worstFVal):
                    self.worstSpiderIndexes.append(i)

        self.fVals = fVals

    def updateBestWorstSpiders(self): # after replacements by offsprings
        """Updates the best and worst spiders of the population."""

        bestFVal = np.inf if self.crit == "min" else -np.inf
        worstFVal = -np.inf if self.crit == "min" else np.inf
        self.bestSpiderIndexes = []
        self.worstSpiderIndexes = []

        for i in range(self.popSize):

            if(self.crit == "min"):

                if (self.fVals[i] < bestFVal):
                    bestFVal = self.fVals[i]
                    self.bestSpiderIndexes = [ i ]

                elif (self.fVals[i] == bestFVal):
                    self.bestSpiderIndexes.append(i)

                if (self.fVals[i] > worstFVal):
                    worstFVal = self.fVals[i]
                    self.worstSpiderIndexes = [ i ]

                elif (self.fVals[i] == worstFVal):
                    self.worstSpiderIndexes.append(i)

            else:

                if (self.fVals[i] > bestFVal):
                    bestFVal = self.fVals[i]
                    self.bestSpiderIndexes = [ i ]

                elif (self.fVals[i] == bestFVal):
                    self.bestSpiderIndexes.append(i)

                if (self.fVals[i] < worstFVal):
                    worstFVal = self.fVals[i]
                    self.worstSpiderIndexes = [ i ]

                elif (self.fVals[i] == worstFVal):
                    self.worstSpiderIndexes.append(i)

    def calculateWeights(self):
        """Calculates all spiders' weights, based on their objective function values.
        Use it after self.calculateFVals."""

        worstFVal = self.fVals[self.worstSpiderIndexes[0]]
        bestFVal = self.fVals[self.bestSpiderIndexes[0]]

        for i in range(self.popSize):

            if(self.crit == "max"): self.weights[i] = (self.fVals[i] - worstFVal) / (bestFVal - worstFVal)
            else: self.weights[i] = (worstFVal - self.fVals[i]) / (worstFVal - bestFVal)

    def distance(self, a, b):
        """Calculates the Euclidean distance between two spiders whose indexes are a and b."""

        if(self.normalizeDistances):
            return np.linalg.norm( self.spiders[a] - self.spiders[b] )/self.diameter

        else:
            return np.log10(1 + np.linalg.norm( self.spiders[a] - self.spiders[b] ))

    def vibc(self, i):
        """Calculates the Vibc vibrations perceived by spider [i] as a result of spider [c]. Returns the result and c's index.
        [c] is the nearest member to [i] which possesses a higher weight than [i]."""

        chosen = -1
        minDistance = np.inf
        vib = 0

        for c in range(self.popSize):

            distance = self.distance(i, c)

            if self.weights[c] > self.weights[i] and distance < minDistance:

                chosen = c
                minDistance = distance

        if(chosen != -1): # found a spider with this criteria.
            vib = self.weights[chosen] * math.exp( -np.power( minDistance, 2 ) )
        # if there wasn't any, vib is 0

        return vib, chosen

    def vibb(self, i):
        """Calculates the Vibc vibrations perceived by spider [i] as a result of the population's best spider."""

        dist = self.distance(i, self.bestSpiderIndexes[0])

        return self.weights[self.bestSpiderIndexes[0]] * math.exp( -np.power( dist, 2 ) )

    def vibf(self, i):
        """Calculates the Vibf vibrations perceived by spider [i] as a result of spider [f]. Returns the result and f's index.
        [f] is the nearest female spider to [i]."""

        chosen = 0
        minDistance = np.inf

        for f in range(self.numFemales):

            distance = self.distance(i, f)

            if distance < minDistance:

                chosen = f
                minDistance = distance

        return self.weights[chosen] * math.exp( -np.power( minDistance, 2 ) ), chosen

    def checkNCorrectBounds(self, spider):
        """Bound checking and correcting function for the genes. If bounds are trespassed,
        the spider is truncated."""

        newSpider = np.copy(spider)

        for i in range( len(newSpider) ):

            if(newSpider[i] < self.bounds[0][i]):
                newSpider[i] = self.bounds[0][i]
                # newSpider[i] = np.random.uniform(self.bounds[0][1], self.bounds[0][1])
                # newSpider[i] = 0

            if(newSpider[i] > self.bounds[1][i]):
                newSpider[i] = self.bounds[1][i]
                # newSpider[i] = np.random.uniform(self.bounds[0][1], self.bounds[0][1])
                # newSpider[i] = 0

        return newSpider

    def updatePositions(self):

        # Female updates (attraction or repulsion)
        for f in range(self.numFemales):

            newSpider = None
            randoms = np.random.uniform(0, 1, 4)
            alpha = randoms[0]
            beta = randoms[1]
            delta = randoms[2]
            rand = randoms[3]
            vibcVal, c = self.vibc(f) # value and index

            draw = np.random.uniform(0, 1)

            if(draw < self.PF):
                newSpider = self.spiders[f] + alpha * vibcVal * (self.spiders[c] - self.spiders[f]) + beta * self.vibb(f) * ( self.spiders[self.bestSpiderIndexes[0]] - self.spiders[f] ) + delta * (rand - 0.5)

            else:

                newSpider = self.spiders[f] - alpha * vibcVal * (self.spiders[c] - self.spiders[f]) - beta * self.vibb(f) * ( self.spiders[self.bestSpiderIndexes[0]] - self.spiders[f] ) + delta * (rand - 0.5)

            if(f != self.bestSpiderIndexes[0]): self.spiders[f] = self.checkNCorrectBounds(newSpider)

        # Discovering dominant and non-dominant males
        self.isDominant = [False for i in range(self.numMales)] # flags for identifying dominants. Resetted at each iteration.
        maleWeights = self.weights[self.numFemales:]
        medianWeight = np.median(maleWeights) # the median is a threshold for identifying dominant males

        # Calculating the weighted mean of the male population (used with non-dominant males)
        maleSum = np.zeros(self.dimensions)

        for m in range(self.numMales): # index by self.numFemales + m

            maleSum += self.spiders[self.numFemales + m] * self.weights[self.numFemales + m]

        maleWeightedMean = maleSum / sum(maleWeights) # returns an array with self.dimensions positions

        for m in range(self.numMales): # index by self.numFemales + m

            if self.weights[self.numFemales + m] > medianWeight: # it's dominant

                self.isDominant[m] = True # preserving flag for the future mating operation

                randoms = np.random.uniform(0, 1, 3)
                alpha = randoms[0]
                delta = randoms[1]
                rand = randoms[2]
                vibfVal, f = self.vibf(self.numFemales + m) # value and index
                newSpider = self.spiders[self.numFemales + m] + alpha * vibfVal * (self.spiders[f] - self.spiders[self.numFemales + m]) + delta * (rand - 0.5)

                if(self.numFemales + m != self.bestSpiderIndexes[0]): self.spiders[self.numFemales + m] = self.checkNCorrectBounds(newSpider)

            else:

                alpha = np.random.uniform(0, 1)
                newSpider = self.spiders[self.numFemales + m] + alpha * maleWeightedMean

                if(self.numFemales + m != self.bestSpiderIndexes[0]): self.spiders[self.numFemales + m] = self.checkNCorrectBounds(newSpider)

    def mating(self):

        for m in range(self.numMales):

            if self.isDominant[m]:

                # checking which females will participate (these are inside the radius)
                femaleIndexes = []

                for f in range(self.numFemales):

                    if( self.distance(self.numFemales + m, f) < self.matingRadius ):
                        femaleIndexes.append(f)

                # perform the mating (roulette algorithm)

                weightSum = self.weights[self.numFemales + m] + sum( np.array( [self.weights[index] for index in femaleIndexes] ) )

                # separating indexes for the roulette
                indexes = [ self.numFemales + m ]
                indexes.extend(femaleIndexes)

                influenceProbabilities = [ self.weights[self.numFemales + m] / weightSum ] # order: male, females
                #initializing with the male's weight

                for index in femaleIndexes:
                    influenceProbabilities.append( self.weights[index]/weightSum )

                slots = [] # stores cumulative probabilities, in order - male, female 1, female 2, ...

                cumProbSum = 0

                for i in range(len(indexes)):
                    cumProbSum += influenceProbabilities[i]
                    slots.append(cumProbSum) # 0.1, 0.2, ..., 1.0

                offspring = np.zeros(self.dimensions)

                for i in range(self.dimensions):

                    draw = np.random.uniform(0, 1)
                    setVar = False # has the variable been changed?

                    for j in range( len(slots) ): # individuals

                        if draw < slots[j]:

                            offspring[i] = self.spiders[ indexes[j] ][i]
                            setVar = True
                            break

                    if not setVar: # accounting for imprecisions which might allow a spider to not have this gene set (1 > 0.999999...)
                        offspring[i] = self.spiders[ indexes[-1] ] # last spider's gene

                # there's a dilemma here: do I recalculate all weights when replacing the worst spider, or only at the next
                # iteration? I've decided to update all weights if an offspring is added.
                weight, fVal = self.offspringWeightFVal(offspring)

                offspringIndex = -1

                # replacing the worst spider by the offspring, if it is better
                # ties between worst or best spiders: choose the first
                if weight > self.weights[self.worstSpiderIndexes[0]]:

                    self.spiders[self.worstSpiderIndexes[0]] = offspring
                    self.weights[self.worstSpiderIndexes[0]] = weight
                    self.fVals[self.worstSpiderIndexes[0]] = fVal
                    offspringIndex = self.worstSpiderIndexes[0]

                    self.updateBestWorstSpiders()
                    self.calculateWeights() # here or after adding all new spiders? Decision: at every new added spider.

                # if it is not better, it is discarded

    def offspringWeightFVal(self, offspring):
        """Calculates an offspring's weight and objective function value.
        Considers that the best and worst spiders have already been found"""

        fVal = self.func(offspring)
        self.FES += 1
        if self.FES == self.maxFES: raise MaxFESReached

        weight = 0
        if(self.crit == "max"): weight = (fVal - self.fVals[self.worstSpiderIndexes[0]]) / (self.fVals[self.bestSpiderIndexes[0]] - self.fVals[self.worstSpiderIndexes[0]])
        else: weight = (self.fVals[self.worstSpiderIndexes[0]] - fVal) / (self.fVals[self.worstSpiderIndexes[0]] - self.fVals[self.bestSpiderIndexes[0]])

        return weight, fVal

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

        Execute after evaluating after using self.calculateFVals or after self.mating!"""

        avg = sum(self.fVals)/self.popSize
        bestVal = self.fVals[self.bestSpiderIndexes[0]]
        bestPoints = [ self.spiders[i] for i in self.bestSpiderIndexes ]
        worstVal = self.fVals[self.worstSpiderIndexes[0]]
        worstPoints = [ self.spiders[i] for i in self.worstSpiderIndexes ]

        error = abs(bestVal - self.optimum)

        return {"avg": avg, "bestVal": bestVal, "bestPoints": bestPoints, "worstVal": worstVal, "worstPoints": worstPoints, "error": error}


    def execute(self):

        self.calculateFVals()
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

        self.calculateWeights()

        try:

            while ( abs(self.fVals[self.bestSpiderIndexes[0]] - self.optimum) > self.tol ):

                self.updatePositions()

                try:
                    self.calculateFVals()

                except MaxFESReached:
                    break

                self.calculateWeights()

                try:
                    self.mating()

                except MaxFESReached:
                    break
                # note that self.mating already updates the spiders' weights after the insertion of new spiders

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

                print(metrics["error"])

        except KeyboardInterrupt:
            return

if __name__ == '__main__':

    # Test of the SSO's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005
    # np.seterr("raise") # any calculation error immediately stops the execution
    dims = 10

    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    SSO = SocialSpiderOptimization(cec2005.F1(dims), bounds, popSize=30, PF=0.7, normalizeDistances=True, optimum=-450) # F5: -310 / others: -450
    #compare normalizing and non-normalizing
    #compare populations of 20, 30 and 50
    SSO.execute()
    results = SSO.results

    print("SSO: for criterion = " + SSO.crit + ", reached optimum of " + str(results["bestFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["bestPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
