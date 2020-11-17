import numpy as np

def getSecond(ind):
    return ind[1]

class MaxFESReached(Exception):
    """Exception used to interrupt the DE operation when the maximum number of fitness evaluations is reached."""
    pass

class DifferentialEvolution(object):
    """Implements a real-valued Differential Evolution."""

    def __init__(self, func, bounds, popSize=100, crit="min", optimum=-450, maxFES=None, tol=1e-08):
        """Initializes the population. Arguments:
        - func: a function name (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - crit: criterion ("min" or "max")
        - optimum: known optimum value for the objective function. Default is 0.
        - maxFES: maximum number of fitness evaluations.
        If set to None, will be calculated as 10000 * [number of dimensions] = 10000 * len(bounds)"""

        # Attribute initialization

        # From arguments
        self.func = func
        self.bounds = bounds
        self.popSize = popSize
        self.crit = crit
        self.optimum = optimum
        self.tol = tol

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * len(bounds[0]) # 10000 x [dimensions]

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
        self.pop = [ [np.random.uniform(self.bounds[0], self.bounds[1]), 0] for i in range(popSize) ] # genes, fitness
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

            if(self.parentSelectionParams): self.parentSelection(*self.parentSelectionParams) # tem parâmetro definido?
            else: self.parentSelection() # se não tiver, roda sem.

            try:
                if(self.crossoverParams): self.crossover(*self.crossoverParams)
                else: self.crossover()

            except MaxFESReached:
                break
                #Exit the loop, going to the result saving part

            try:
                for index in range( len(self.children) ):
                    if(self.mutationParams): self.mutation(index, *self.mutationParams)
                    else: self.mutation(index)

            except MaxFESReached:
                break

            if(self.newPopSelectionParams): self.newPopSelection(*self.newPopSelectionParams)
            else: self.newPopSelection()

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

    def getMax(self):
        """Finds the individuals with the highest fitness value of the population.
        Returns (top, points) -> top = fitness value / points: list of the individuals' genes.
        Execute after evaluating fitness values for the entire population!"""

        top = -np.inf
        points = []

        for i in range(self.popSize):

            if (top < self.pop[i][1]):
                top = self.pop[i][1]
                points = [ self.pop[i][0] ]

            elif (top == self.pop[i][1]):
                points.append(self.pop[i][0])

        if(self.crit == "max"): self.bestSoFar = top

        return (top, points)

    def getMin(self):
        """Finds the individuals with the lowest fitness value of the population.
        Returns (bottom, points) -> bottom = fitness value / points: list of the individuals' genes.
        Execute after evaluating fitness values for the entire population!"""

        bottom = np.inf
        points = []

        for i in range(self.popSize):

            if (bottom > self.pop[i][1]):
                bottom = self.pop[i][1]
                points = [ self.pop[i][0] ]

            elif (bottom == self.pop[i][1]):
                points.append(self.pop[i][0])

        if(self.crit == "min"): self.bestSoFar = bottom

        return (bottom, points)

    def getMean(self):
        """Returns the population's mean fitness value. Execute after evaluating fitness values for the entire population!"""

        total = 0

        for i in range(self.popSize):

            total += self.pop[i][1]

        return total/self.popSize

    def getFitnessMetrics(self):

        """Finds the mean, greater and lower fitness values for the population,
        as well as the points with the greater and lower ones.
        Returns a dict, whose keys are:
        "avg" to average value
        "top" to top value
        "topPoints" to a list of points with the top value
        "bottom" to bottom value
        "bottomPoints" to a list of points with the bottom value

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

    def mutation(self, base="rand", F=1, nDiffs=1):
        """Executes the mutation procedure for the entire population. Parameters:
        - base: string that identifies who is the perturbed vector for mutation. "rand" (default) choses random vectors;
        "best" choses the population's best solution.
        - F ∈ [0, 2]; defaults to 1: affects the mutation strength;
        - nDiffs (integer, defaults to 1): number of employed difference vector"""

        self.mutedPop = []

        for i in range(self.popSize):

            selectedIndexes = []

            x = None
            perturbation = np.zeros( len( self.bounds[0] ) )
            # numpy array: [0 0 ... 0] - size = number of dimensions

            if base == "rand":
                xIndex = np.random.randint(0, self.popSize)
                x = self.pop[xIndex]
                selectedIndexes.append(xIndex)

            elif base == "best":
                x = self.pop[self.bestIndex]
                selectedIndexes.append(self.bestIndex)

            else:
                raise ValueError("DifferentialEvolution: mutation: best expects values 'rand' or 'best'.")

            for i in range(nDiffs):

                ind1, ind2 = None, None
                index1, index2 = -1, -1

                # Choosing vectors that were not already chosen
                while true:

                    index1 = np.random.randint(0, self.popSize)

                    if(index1 not in selectedIndexes):
                        ind1 = self.pop[index1]
                        selectedIndexes.append(index1)
                        break


                while true:

                    index2 = np.random.randint(0, self.popSize)

                    if(index2 not in selectedIndexes):
                        ind2 = self.pop[index2]
                        selectedIndexes.append(index2)
                        break

            perturbation += F * (ind1 - ind2)

            self.mutedPop.append()


        while True:
            # adds a random value to the gene with a probability prob
            newGenes = [ gene + np.random.normal(mean, stdev) if (np.random.uniform(0, 1) < prob) else gene for gene in self.children[index][0] ]

            #redo bound check
            if( self.isInBounds([newGenes, 0]) ):

                if(self.pop[index][0] != newGenes):

                    self.pop[index][0] = newGenes
                    self.calculateFitnessInd(index)

                break

    def isInBounds(self, ind):
        """Bound checking function for the genes. Used for mutation and crossover."""

        for i in range( len(ind[0]) ):

            if not (self.bounds[0][i] <= ind[0][i] <= self.bounds[1][i]): return False
            # if this gene is in the bounds, inBounds keeps its True value.
            # else, it automatically returns False. Escaping to save up iterations.

        return True # if it has exited the loop, the genes are valid

    def getElite(self):

        if self.eliteSize > 0:

            elite = None

            if self.crit == "max":
                self.pop.sort(key = getSecond, reverse = True) # ordena pelo valor de f em ordem decrescente
                elite = self.pop[:self.eliteSize]

            else:
                self.pop.sort(key = getSecond, reverse = False) # ordena pelo valor de f em ordem crescente
                elite = self.pop[:self.eliteSize]

            self.elite = elite

    def tournamentSelection(self, crossover = True):
        # use with matingPool for parent selection
        # use with pop for post-crossover selection (non-generational selection schemes)

        winners = []

        if(not crossover and self.elite): # if there is an elite and it is not a crossover selection...
            for ind in self.elite:
                winners.extend(self.elite)

        while len(winners) < self.popSize:

            positions = np.random.randint(0, len(self.pop), 2)
            # len(self.pop) because the population may have children (larger than self.popSize)
            ind1, ind2 = self.pop[positions[0]], self.pop[positions[1]]
            if self.crit == "min": winner = ind1 if ind1[1] <= ind2[1] else ind2 # compara valores de f. escolhe o de menor aptidão
            else: winner = ind1 if ind1[1] >= ind2[1] else ind2 # compara valores de f. escolhe o de menor aptidão
            winners.append(winner)

        if crossover: self.matingPool = winners
        else: self.pop = winners # post-crossover selection determines the population

    def blxAlphaCrossover(self, alpha=0.5, crossProb=0.6):
        # Defines the BLX-α crossover for the mating pool. Creates an amount of children equal to the population size.
        if not self.matingPool:
            raise ValueError("There is no mating pool. Execute a selection function for it first.")

        children = []

        for i in range(self.popSize):

            positions = np.random.randint(0, len(self.matingPool), 2)
            parent1, parent2 = self.matingPool[positions[0]], self.matingPool[positions[1]]

            if (np.random.uniform(0, 1) < crossProb): # crossover executed with probability crossProb

                child = []
                genes = []

                for j in range( len(parent1[0]) ): # iterate through its genes

                    while True:
                        beta = ( np.random.uniform( -alpha, 1 + alpha ) )
                        gene = parent1[0][j] + beta * (parent2[0][j] - parent1[0][j])

                        if( self.bounds[0][j] <= gene <= self.bounds[1][j] ):
                            genes.append(gene)
                            break
                            #Fora dos limites? Refazer.

                child.append(genes)
                child.append(self.func(genes))
                self.FES += 1
                if self.FES == self.maxFES: raise MaxFESReached

                children.append(child)

            else: #if it is not executed, the parent with the best fitness is given as a child
                if self.crit == "min": children.append(parent1) if parent1[1] <= parent2[1] else children.append(parent2) # compara valores de f. escolhe o de menor aptidão
                else: children.append(parent1) if parent1[1] >= parent2[1] else children.append(parent2) # compara valores de f. escolhe o de menor aptidão

        self.children = children

        if(self.newPopSelection != self.generationalSelection): self.pop.extend(self.children)

    def generationalSelection(self):

        if self.eliteSize > 0:

            if self.crit == "max":
                self.children.sort(key = getSecond, reverse = True) # ordena pelo valor de f em ordem decrescente

            else:
                self.children.sort(key = getSecond, reverse = False) # ordena pelo valor de f em ordem crescente

            newPop = []
            newPop.extend(self.elite)
            newPop.extend(self.children)
            self.pop = newPop[:self.popSize] # cutting out the worst individuals

        else:
            self.pop = self.children


if __name__ == '__main__':

    # Test of the GA's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005

    bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    GA = GeneticAlgorithm(cec2005.F1(10), bounds, eliteSize=1, popSize=50)

    GA.setParentSelection(GA.tournamentSelection, (True,) )
    GA.setCrossover(GA.blxAlphaCrossover, (0.5, 0.6)) # alpha, prob
    GA.setMutation(GA.creepMutation, (0.05, 0, 1)) # prob, mean, sigma
    GA.setNewPopSelection(GA.tournamentSelection, (False, ))
    # GA.setNewPopSelection(GA.generationalSelection, None)
    GA.execute()
    results = GA.results

    print("GA: for criterion = " + GA.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["minPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
