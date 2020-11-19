import numpy as np

def getSecond(ind):
    return ind[1]

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

class DifferentialEvolution(object):
    """Implements a real-valued Differential Evolution."""

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
        self.pop = None
        self.mutedPop = None
        self.crossedPop = None
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.bestSoFar = 0
        self.bestIndex = 0 # index of the best individual in the population
        self.mutation = None
        self.mutationParams = None
        self.results = None

        if( len(self.bounds[0]) != len(self.bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        # Population initialization as random (uniform)
        self.pop = [ [np.random.uniform(self.bounds[0], self.bounds[1]), 0] for i in range(self.popSize) ] # genes, fitness
        self.calculateFitnessPop()
        # individuals' parameters are numpy arrays

    def setCrossover(self, crossover, crossoverParams):
        """Configure the used mutation process. Parameters:
        - crossover: a crossover function
        - crossoverParams: its parameters (a tuple)"""
        self.crossover = crossover
        self.crossoverParams = crossoverParams

    def setMutation(self, mutation, mutationParams):
        """Configure the used mutation process. Parameters:
        - mutation: a mutation function
        - mutationParams: its parameters (a tuple)
        (Keep in mind that mutation functions also require an (integer) individual's index before the params)"""
        self.mutation = mutation
        self.mutationParams = mutationParams

    def execute(self):

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

            try:

                if(self.mutationParams): self.mutation(*self.mutationParams) # tem parâmetro definido?
                else: self.mutation() # se não tiver, roda sem.

            except MaxFESReached:
                break

            try:

                if(self.crossoverParams): self.crossover(*self.crossoverParams)
                else: self.crossover()

            except MaxFESReached:
                break
                #Exit the loop, going to the result saving part

            self.selection()

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

    def isInBounds(self, ind):
        """Bound checking function for the genes. Used in the mutation procedure."""

        for i in range( len(ind[0]) ):

            if not (self.bounds[0][i] <= ind[0][i] <= self.bounds[1][i]): return False
            # if one gene is out of the bounds, returns false

        return True # if it has exited the loop, the genes are valid

    def calculateFitnessPop(self):
        """Calculates the fitness values for the entire population."""

        for ind in self.pop:
            ind[1] = self.func(ind[0])
            self.FES += 1

            if self.FES == self.maxFES: raise MaxFESReached

    def calculateFitnessInd(self, index):
        """Calculates the fitness values for a specific element (specified by index)."""

        self.pop[index][1] = self.func(self.pop[index][0])
        self.FES += 1

        if self.FES == self.maxFES: raise MaxFESReached

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

            total += self.pop[i][1]

            if (top < self.pop[i][1]):
                top = self.pop[i][1]
                topPoints = [ self.pop[i][0] ]

            elif (top == self.pop[i][1]):
                topPoints.append(self.pop[i][0])

            if (bottom > self.pop[i][1]):
                bottom = self.pop[i][1]
                bottomPoints = [ self.pop[i][0] ]

            elif (bottom == self.pop[i][1]):
                bottomPoints.append(self.pop[i][0])

        avg = total/self.popSize

        if(self.crit == "min"): self.bestSoFar = bottom
        if(self.crit == "max"): self.bestSoFar = top

        error = abs(self.bestSoFar - self.optimum)

        return {"avg": avg, "top": top, "topPoints": topPoints, "bottom": bottom, "bottomPoints": bottomPoints, "error": error}

    def classicMutation(self, base="rand", F=0.5, nDiffs=1):
        """Executes the mutation procedure for the entire population. Parameters:
        - base: string that identifies who is the perturbed vector for mutation. "rand" (default) choses random vectors;
        "best" choses the population's best solution.
        "current to best adds a perturbation that is the distance between the current individual and the best one"
        - F ∈ [0, 2]; defaults to 1: affects the mutation strength;
        - nDiffs (integer, defaults to 1): number of employed difference vector"""

        self.mutedPop = []

        for i in range(self.popSize):

            while True:

                selectedIndexes = []

                x = None
                perturbation = np.zeros(self.dimensions)
                # numpy array: [0 0 ... 0] - size = number of dimensions

                if base == "rand":
                    xIndex = np.random.randint(0, self.popSize)
                    x = self.pop[xIndex]
                    selectedIndexes.append(xIndex)

                elif base == "best":
                    x = self.pop[self.bestIndex]
                    selectedIndexes.append(self.bestIndex)

                elif base == "current-to-best":
                    x = self.pop[i]
                    best = self.pop[self.bestIndex]
                    selectedIndexes.append(i)
                    selectedIndexes.append(self.bestIndex)

                else:
                    raise ValueError("DifferentialEvolution: mutation: best expects values 'rand', 'best' or 'current-to-best'.")

                for j in range(nDiffs):

                    ind1, ind2 = None, None
                    index1, index2 = -1, -1

                    # Choosing vectors that were not already chosen
                    while True:

                        index1 = np.random.randint(0, self.popSize)

                        if(index1 not in selectedIndexes):
                            ind1 = self.pop[index1]
                            selectedIndexes.append(index1)
                            break

                    while True:

                        index2 = np.random.randint(0, self.popSize)

                        if(index2 not in selectedIndexes):
                            ind2 = self.pop[index2]
                            selectedIndexes.append(index2)
                            break

                perturbation += F * (ind1[0] - ind2[0]) # ind[0] carries its genes

                if base == "current-to-best":

                    perturbation += F * (best[0] - x[0])

                v = [x[0] + perturbation, 0]

                if(self.isInBounds(v)):

                    v[1] = self.func(v[0])
                    self.FES += 1
                    if self.FES == self.maxFES: raise MaxFESReached
                    self.mutedPop.append(v) # x[0]: genes / second element: fitness value

                    break


    def classicCrossover(self, type="bin", cr=0.8):
        """Defines the crossover, which forms the trial vectors. Its parameters are:
        - cr (int ∈ [0, 1]), is the crossover constant, which regulates the proportion
        of genes passed by each parent;
        - type: string (defaults to 'bin'): the type of crossover procedure. Currently, only 'bin',
        corresponding to independent binomial experiments is supported."""

        self.crossedPop = []

        if(type == "bin"):

            for i in range(self.popSize): # iterating through individuals

                crossedInd = [ np.zeros(self.dimensions), 0]
                randDim = np.random.randint(0, self.dimensions) # a random position WILL receive a muted parameter

                for j in range(self.dimensions): # iterating through dimensions

                    if (np.random.uniform(0, 1) < cr or j == randDim):
                        # crossover executed with probability crossProb
                        # if the current dimension is randDim, this position receives the muted gen

                        crossedInd[0][j] = self.mutedPop[i][0][j] # mutedPop indexes: individual, genes, dimension

                    else:
                        crossedInd[0][j] = self.pop[i][0][j]

                crossedInd[1] = self.func(crossedInd[0])
                self.FES += 1
                if self.FES == self.maxFES: raise MaxFESReached
                self.crossedPop.append(crossedInd)

    # def getElite(self):
    #
    #     if self.eliteSize > 0:
    #
    #         elite = None
    #
    #         if self.crit == "max":
    #             self.pop.sort(key = getSecond, reverse = True) # ordena pelo valor de f em ordem decrescente
    #             elite = self.pop[:self.eliteSize]
    #
    #         else:
    #             self.pop.sort(key = getSecond, reverse = False) # ordena pelo valor de f em ordem crescente
    #             elite = self.pop[:self.eliteSize]
    #
    #         self.elite = elite

    def selection(self):
        """Defines the selection procedure for DEs."""

        finalPop = []

        for i in range(self.popSize):

            finalInd = None

            if self.crit == "min": finalInd = self.mutedPop[i] if self.mutedPop[i][1] <= self.pop[i][1] else self.pop[i] # compara valores de f. escolhe o de menor aptidão
            else: finalInd = self.mutedPop[i] if self.mutedPop[i][1] >= self.pop[i][1] else self.pop[i] # compara valores de f. escolhe o de menor aptidão

            finalPop.append(finalInd)

        self.pop = finalPop


if __name__ == '__main__':

    # Test of the DE's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005

    bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    DE = DifferentialEvolution(cec2005.F1(10), bounds)
    DE.setMutation(DE.classicMutation, ("rand", 1, 1)) # base, F, nDiffs
    DE.setCrossover(DE.classicCrossover, ("bin", 0.5)) # type, CR
    DE.execute()
    results = DE.results

    print("DE: for criterion = " + DE.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["minPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
