import numpy as np

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

class RegPSO(object):
    """Implements a real-valued Particle Swarm Optimization."""

    def __init__(self, func, bounds, popSize=None, globalWeight=2.05,
    localWeight=2.05, clerkK=False, inertiaDecay=True, prematureThreshold=1.1e-04,
    crit="min", optimum=-450, maxFES=None, tol=1e-08):
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
        self.clerkK = clerkK
        self.inertiaDecay = inertiaDecay
        self.prematureThreshold = prematureThreshold
        self.regroupingFactor = 6/(5*self.prematureThreshold)

        if(popSize): self.popSize = popSize
        else: self.popSize = 10 * self.dimensions

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * self.dimensions

        # self.vMax = [ (self.bounds[1][i] - self.bounds[0][i])/2 for i in range( len( self.bounds[0] ) ) ]
        self.vMax = [ self.dimensions for i in range( len( self.bounds[0] ) ) ]

        self.globalWeight = globalWeight
        self.localWeight = localWeight
        self.initialInertiaWeight = 0.9
        self.finalInertiaWeight = 0.4
        self.inertiaWeight = self.initialInertiaWeight # weight modified by decay
        phi = self.localWeight + self.globalWeight
        self.K = abs( 2 / ( phi - 2 + np.sqrt( np.power(phi, 2) - 4*phi ) ) ) if phi > 4 and self.clerkK else 1 # Clerc's constriction factor

        # Control attributes
        self.positions = []
        self.velocities = []
        self.fVals = None
        self.pBest = None
        self.pBestFVals = None
        self.gBestIndex = 0
        self.gBestValue = np.inf if crit == "min" else -np.inf
        self.swarmRadius = 0 # maximum Euclidean distance from the global best
        self.originalRanges = np.array([ abs(self.bounds[0][i] - self.bounds[1][i]) for i in range (self.dimensions) ])
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.bestSoFar = 0
        self.bestIndex = 0 # index of the best individual in the population
        self.results = None

        # Population initialization as random (uniform)
        for i in range(self.popSize):

            self.positions.append( np.random.uniform(self.bounds[0], self.bounds[1]) )

        self.positions = np.array(self.positions) # result: matrix. Lines are individuals; columns are dimensions

        # Initializing speeds as random between maximum and minimum values
        for i in range(self.popSize):

            self.velocities.append( [np.random.uniform(-self.vMax[j], self.vMax[j]) for j in range(self.dimensions)] )


        self.velocities = np.array(self.velocities)
        # self.velocities = np.zeros((self.popSize, self.dimensions))


        self.fVals = np.zeros(self.popSize)
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

        try:

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

                self.calculateSwarmRadius()

                if self.swarmRadius < self.prematureThreshold:
                    self.regroup()

        except KeyboardInterrupt:
            return

    def calculateFitnessPop(self):
        """Calculates the fitness values for the entire population, and updates personal and group best values."""

        self.swarmRadius = -np.inf # maximum Euclidean distance from the global best

        for i in range(self.popSize):
            # Fitness calculations
            self.fVals[i] = self.func(self.positions[i])
            self.FES += 1
            if self.FES == self.maxFES: raise MaxFESReached

            if(self.crit == "min"):

                if(self.fVals[i] < self.pBestFVals[i]):
                    self.pBest[i] = self.positions[i]
                    self.pBestFVals[i] = self.fVals[i]

                if(self.fVals[i] < self.gBestValue):
                    self.gBestIndex = i
                    self.gBestValue = self.fVals[i]

            else:

                if(self.fVals[i] > self.pBestFVals[i]):
                    self.pBest[i] = self.positions[i]
                    self.pBestFVals[i] = self.fVals[i]

                if(self.fVals[i] > self.gBestValue):
                    self.gBestIndex = i
                    self.gBestValue = self.fVals[i]


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

        avg = total/self.popSize

        if(self.crit == "min"): self.bestSoFar = bottom
        if(self.crit == "max"): self.bestSoFar = top

        error = abs(self.bestSoFar - self.optimum)

        return {"avg": avg, "top": top, "topPoints": topPoints, "bottom": bottom, "bottomPoints": bottomPoints, "error": error}

    def calculateNewSpeeds(self):

        if(self.inertiaDecay):
            self.inertiaWeight = ( (self.maxFES - self.FES)/self.maxFES ) * (self.initialInertiaWeight - self.finalInertiaWeight) + self.finalInertiaWeight

        for i in range(self.popSize):
            self.velocities[i] = self.K * ( self.inertiaWeight * self.velocities[i] + # inertia
            self.localWeight * np.random.uniform(0, 1) * (self.pBest[i] - self.positions[i]) + # local "nostalgia"
            self.globalWeight * np.random.uniform(0, 1) * (self.pBest[self.gBestIndex] - self.positions[i]) ) # global knowledge

            # comparing speeds with speed limits at each dimension

            for j in range(self.dimensions):

                if(self.velocities[i][j] > self.vMax[j]):
                    self.velocities[i][j] = self.vMax[j]

                if(self.velocities[i][j] < -self.vMax[j]):
                    self.velocities[i][j] = -self.vMax[j]

    def updatePositions(self):

        for i in range(self.popSize): # population

            self.positions[i] += self.velocities[i]

            # bound checking: if the bounds are trespassed, the variables are truncated (commented: reset as random)
            for j in range(self.dimensions): # dimension

                if(self.positions[i][j] < self.bounds[0][j]):
                    # self.positions[i][j] = self.bounds[0][j]
                    self.positions[i][j] = np.random.uniform(self.bounds[0][j], self.bounds[1][j])

                if(self.positions[i][j] > self.bounds[1][j]):
                    # self.positions[i][j] = self.bounds[1][j]
                    self.positions[i][j] = np.random.uniform(self.bounds[0][j], self.bounds[1][j])

    def calculateSwarmRadius(self):

        for i in range(self.popSize):

            dist = np.linalg.norm( self.positions[i] - self.pBest[self.gBestIndex] )

            if dist > self.swarmRadius:
                self.swarmRadius = dist

        self.swarmRadius /= np.linalg.norm( np.array(self.bounds[1]) - np.array(self.bounds[0]) ) # normalizing by the diameter

    def regroup(self):
        # defining the new range (remember that the original range - self.originalRanges - is calculated
        # on __init__)

        # checking the maximum distance to the global best for each dimension

        maxDists = np.array([-np.inf for i in range(self.dimensions)])

        for i in range(self.popSize):
            for j in range(self.dimensions):
                if (self.positions[i][j] - self.pBest[self.gBestIndex][j] > maxDists[j]):
                    maxDists[j] = self.positions[i][j] - self.pBest[self.gBestIndex][j]

        newRange = self.regroupingFactor * maxDists

        for i in range(self.popSize):

            randomVec = np.zeros(self.dimensions)

            for j in range(self.dimensions):

                randomVec[j] = np.random.uniform(-newRange[j], +newRange[j])

            self.positions[i] = self.pBest[self.gBestIndex] + randomVec

        self.vMax = 0.05 * newRange


if __name__ == '__main__':

    # Test of the PSO's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005

    bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    PSO = RegPSO(cec2005.F4(10), bounds, popSize=50, clerkK=False, inertiaDecay=True)
    PSO.execute()
    results = PSO.results

    print("PSO: for criterion = " + PSO.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["minPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
