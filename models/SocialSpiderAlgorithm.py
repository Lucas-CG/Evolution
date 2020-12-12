import numpy as np
import math
import time
from scipy.spatial.distance import pdist, squareform

class SocialSpiderAlgorithm(object):

    def __init__(self, func, bounds, popSize=None, crit="min", popSize=30, vibrationConstant=-700, attenuationRate=1, maskChangeProb=0.5, maskOneProb=0.5, optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - vibrationConstant: a constant used for calculating vibrations. Has to be lower than all possible values.
        - attenuation rate: real-valued parameter used to regulate the intensity of vibrations. Greater values reduce the vibrations.
        - maskChangeProb: probability of not changing a spider's dimension mask at its movement phase.
        - maskOneProb: if the mask will be changed, probability that a bit will become 1.
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

        self.attenuationRate = attenuationRate
        self.maskChangeProb = maskChangeProb
        self.maskOneProb = maskOneProb


        # Control attributes
        self.spiders = []
        self.fVals = np.zeros(self.popSize)
        self.targetVibrations = np.zeros(self.popSize) # target vibration intensities per spider
        self.targetPositions = np.zeros(shape=(self.popSize, self.dimensions)) # target positions per spider
        self.iterationsSinceLastChange = np.zeros(self.popSize)
        self.pastMovements = np.zeros( shape=(self.popSize, self.dimensions) )
        self.masks = [ [False for i in range(self.dimensions)] for i in range(self.popSize) ]
        self.vibrations = np.zeros( shape=(self.popSize, self.popSize) )
        self.distances = np.zeros( shape=(self.popSize, self.popSize) )
        self.stdMean = 0
        self.bestSpiderIndexes = []
        self.worstSpiderIndexes = []
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.results = None


    def randomSolution(self):
        return np.random.uniform(self.bounds[0], self.bounds[1])

    def execute(self):

        # Spiders initialization as random (uniform)
        self.spiders = np.random.uniform(bounds[0], bounds[1], size=(self.popSize, self.dimensions))
        # result: matrix. Lines are individuals; columns are dimensions
        self.targets = np.copy(self.spiders)

        # Arrays for collecting metrics

        generations = []
        FESCount = []
        errors = []
        bestFits = []
        bestPoints = []
        worstFits = []
        worstPoints = []
        avgFits = []

        try:

            while ( abs(self.fVals[self.bestSpiderIndexes[0]] - self.optimum) > self.tol ):

                try:
                    self.calculateFVals()

                except MaxFESReached:
                    break

                metrics = self.getFitnessMetrics()

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

                self.calcDistances()
                self.generateVibrations()
                self.moveSpiders()

                self.genCount += 1


        except KeyboardInterrupt:
            return


    def vibration(self, index1, index2):
        """Calculates a vibration sent from self.spiders[index1] to self.spiders[index2]. If index1 and index2 are equal,
        calculates the vibration generated from self.spiders[index1]. Supposes that self.calcDistances has already been
        executed."""

        if index1 == index2:

            if(self.crit == "min"):
                return np.log10( ( 1 / (self.fVals[index1] - self.vibrationConstant) ) + 1 )

            else:
                return np.log10(self.fVals[index1])

        else:

            return self.vibrations[index1][index1] * np.exp( -(self.distances[index1][index2]) / (self.stdMean * self.attenuationRate) )

    def generateVibrations(self):
        """Generates the vibrations for every pair of spiders."""

        # "self vibrations"
        for i in range(self.popSize):
            self.vibrations[i][i] = self.vibration(i, i)

        # vibrations between spiders
        for i in range(self.popSize):
            for j in range(self.popSize):
                if(i != j): self.vibrations[i][j] = self.vibration(i, j) # i: source; j: destination

    def calcStdMean(self):
        """Calculates the mean of the standard deviations at every dimension."""
        return base_distance = np.mean(np.std(self.spiders, 0))

    def calcDistances(self):
        """Calculates the Manhattan distances between all pairs of spiders. Saves it to self.distances."""
        self.distances = squareform(pdist(self.spiders, 'cityblock'))

    def moveSpiders(self):

        maxVibrationIntensities = []
        maxVibrationOrigins = []

        # finding which is the strongest max vibration and its source
        for j in range(self.popSize):

            maxVib = -np.inf
            maxInd = -1

            for i in range(self.popSize):

                if(j != i):
                    if(self.vibrations[i][j] > maxVib):
                        maxVib = self.vibrations[i][j]
                        maxInd = i

            maxVibrationIntensities.append(maxVib)
            maxVibrationOrigins.append(maxInd)

        for i in range(self.popSize):

            # checking if the strongest vibrations are stronger than the past targets
            # if they are, change the target and reset the counter.
            # if they aren't, just increment the counter
            if(maxVibrationIntensities[i] > self.targets[i]):
                self.targetVibrations[i] = maxVibrationIntensities[i]
                self.targetPositions[i] = maxVibrationOrigins[i]
                self.iterationsSinceLastChange[i] = 0

            else:
                self.iterationsSinceLastChange[i] += 1

            # checking if the mask will be changed
            draw = np.random.uniform(0, 1)

            if draw < 1 - ((self.maskChangeProb) ** self.iterationsSinceLastChange[i]) :
                # change the mask
                for j in range(self.dimensions):
                    draw = np.random.uniform(0, 1)
                    self.masks[i] = True if draw < self.maskOneProb else False

                if( max(self.masks[i]) == False ): # every bit is 0
                    position = np.random.randint(0, self.dimensions)
                    self.masks[i][position] = True

                if( min(self.masks[i]) == True ): # every bit is 1
                    position = np.random.randint(0, self.dimensions)
                    self.masks[i][position] = False


            pFollowing = []

            for j in range(self.dimensions):

                if self.masks[i][j] == False: # 0 bit
                    pFollowing.append(self.targetPositions[i][j])

                else: # 1 bit
                    randomInd = np.random.randint(0, self.popSize)
                    pFollowing.append(self.spiders[randomInd][j])

            pFollowing = np.array(pFollowing)

            movement = np.random.uniform(0, 1) * self.pastMovements[i] +
                       np.multiply( (self.pFollowing - self.spiders[i]), np.random.uniform(0, 1, shape=(self.dimensions, ) ) )
            # note: np.multiply = element-wise multiplication
            self.spiders[i] = self.spiders[i] + movement
            self.pastMovements[i] = movement





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


if __name__ == '__main__':
    pass
