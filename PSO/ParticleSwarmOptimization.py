import numpy as np

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

class ParticleSwarmOptimization(object):
    """Implements a real-valued Particle Swarm Optimization."""

    def __init__(self, func, bounds, popSize=None, crit="min", optimum=-450, maxFES=None, tol=1e-08):
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
        self.positions = []
        self.velocities = None
        self.fVals = None
        self.pBest = None
        self.pBestFVals = None
        self.gBestIndex = 0
        self.gBestValue = np.inf if crit == "min" else -np.inf
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.bestSoFar = 0
        self.bestIndex = 0 # index of the best individual in the population
        self.results = None

        if( len(self.bounds[0]) != len(self.bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        # Population initialization as random (uniform)
        for i in range(self.popSize):

            self.positions.append( np.random.uniform(self.bounds[0], self.bounds[1]) )

        self.positions = np.array(self.positions) # result: matrix. Lines are individuals; columns are dimensions
        self.velocities = np.zeros((self.popSize, self.dimensions))
        self.fVals = np.array([0 for i in range(self.popSize)]) if crit == "min" else np.array([0 for i in range(self.popSize)])
        self.pBest = np.zeros((self.popSize, self.dimensions))
        self.pBestFVals = np.array([np.inf for i in range(self.popSize)]) if crit == "min" else np.array([-np.inf for i in range(self.popSize)])

    def execute(self):

        self.calculateFitnessPop()
        metrics = self.getFitnessMetrics() # post-initialization: generation 0

        # Arrays for collecting metrics

        generations = [ self.genCount ]
        FESCount = [ self.FES ]
        errors = [ metrics["error"] ]
        maxFits = [ metrics["top"] ]
        maxPoints = [ metrics["topPoints"] ]
        minFits = [ metrics["bottom"] ]
        minPoints = [ metrics["bottomPoints"] ]
        avgFits = [ metrics["avg"] ]

        while ( abs(self.bestSoFar - self.optimum) > self.tol ):

            self.calculateNewSpeeds()
            self.updatePositions()

            try:
                self.calculateFitnessPop()

            except MaxFESReached:
                break

            metrics = self.getFitnessMetrics()

            self.genCount += 1

            generations.append(self.genCount)
            FESCount.append(self.FES)
            errors.append(metrics["error"])
            maxFits.append(metrics["top"])
            maxPoints.append(metrics["topPoints"])
            minFits.append(metrics["bottom"])
            minPoints.append(metrics["bottomPoints"])
            avgFits.append(metrics["avg"])

            self.results = {"generations": generations,
                "FESCounts": FESCount,
                "errors": errors,
                "maxFits": maxFits,
                "maxPoints": maxPoints,
                "minFits": minFits,
                "minPoints": minPoints,
                "avgFits": avgFits}

    def calculateFitnessPop(self):
        """Calculates the fitness values for the entire population, and updates personal and group best values."""

        for i in range(self.popSize):
            # Fitness calculations
            self.fVals[i] = self.func(self.positions[i])
            self.FES += 1
            if self.FES == self.maxFES: raise MaxFESReached

            if(self.crit == "min"):

                if(self.fVals[i] < self.pBestFVals[i]):
                    self.pBest[i] = self.positions[i]
                    self.pBestFVals[i] = self.fVals[i]

            else:

                if(self.fVals[i] > self.pBestFVals[i]):
                    self.pBest[i] = self.positions[i]
                    self.pBestFVals[i] = self.fVals[i]


    def getFitnessMetrics(self):

        """Finds the mean, greater and lower fitness values for the population,
        as well as the points with the greater and lower ones and the current error.
        Returns a dict, whose keys are:
        "avg" to average value
        "top" to top value
        "topPoints" to a list of points with the top value
        "bottom" to bottom value
        "bottomPoints" to a list of points with the bottom value
        "error" for the current error (difference between the fitness and the optimum)

        Execute after evaluating fitness values for the entire population!"""

        total = 0
        top = -np.inf
        topPoints = []
        bottom = np.inf
        bottomPoints = []

        for i in range(self.popSize):

            total += self.fVals[i]

            if (top < self.fVals[i]):
                top = self.fVals[i]
                topPoints = [ self.positions[i] ]

            elif (top == self.fVals[i]):
                topPoints.append(self.positions[i])

            if (bottom > self.fVals[i]):
                bottom = self.fVals[i]
                bottomPoints = [ self.positions[i] ]

            elif (bottom == self.fVals[i]):
                bottomPoints.append(self.positions[i])

            if(self.crit == "min"):

                if(self.fVals[i] < self.gBestValue):
                    self.gBestIndex = i
                    self.gBestValue = self.fVals[i]

            else:

                if(self.fVals[i] > self.gBestValue):
                    self.gBestIndex = i
                    self.gBestValue = self.fVals[í]

        avg = total/self.popSize

        if(self.crit == "min"): self.bestSoFar = bottom
        if(self.crit == "max"): self.bestSoFar = top

        error = abs(self.bestSoFar - self.optimum)

        return {"avg": avg, "top": top, "topPoints": topPoints, "bottom": bottom, "bottomPoints": bottomPoints, "error": error}

    def calculateNewSpeeds(self):

        for i in range(self.popSize):
            self.velocities[i] += 2 * np.random.uniform(0, 1) * (self.pBest[i] - self.positions[i]) # personal nostalgia
            self.velocities[i] += 2 * np.random.uniform(0, 1) * (self.pBest[self.gBestIndex] - self.positions[i]) # global knowledge

    def updatePositions(self):

        for i in range(self.popSize): # population

            self.positions[i] += self.velocities[i]

            # bound checking: if the bounds are trespassed, the variables are truncated (commented: reset as random)
            for j in range(self.dimensions): # dimension

                if(self.positions[i][j] < self.bounds[0][j]):
                    self.positions[i][j] = self.bounds[0][j]
                    # self.positions[i][j] = np.random.uniform(self.bounds[0][j], self.bounds[1][j])

                if(self.positions[i][j] > self.bounds[1][j]):
                    self.positions[i][j] = self.bounds[1][j]
                    # self.positions[i][j] = np.random.uniform(self.bounds[0][j], self.bounds[1][j])

if __name__ == '__main__':

    # Test of the PSO's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005

    bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    PSO = ParticleSwarmOptimization(cec2005.F1(10), bounds)
    PSO.execute()
    results = PSO.results

    print("PSO: for criterion = " + PSO.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["minPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
