import numpy as np

class MaxFESReached(Exception):
    """Exception used to interrupt the GA operation when the maximum number of fitness evaluations is reached."""
    pass

class CMAES(object):
    """Implements a real-valued CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, func, bounds, crit="min", eliteSize=0, mu=None, lamb=None, equalWeights=False, cc=None, ccov=None, csigma=None, dsigma=None, optimum=0, maxFES=None, tol=1e-08):
        """Initializes the algorithm. Arguments:
        - func: a function name (the optimization problem to be resolved).
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - crit: criterion ("min" or "max").
        - eliteSize: positive integer; defines whether elitism is enabled or not and the elite's size.
        - mu: parent number: the best individuals selected at each generation.
        - lamb: population size: generated individuals, which will be selected.
        - equalWeights: bool: defines whether "stronger" individuals will have greater weights at the crossover.
        - cc: present path's weight. Its inverse indicates this information's duration on the adaptation procedure.
        - ccov: present covariance matrix's weight. Its inverse indicates this information's duration on the adaptation procedure.
        - csigma: present sigma/step size's weight. Its inverse indicates this information's duration on the adaptation procedure.
        - dsigma: damping constant for adaptation of step sizes.
        - optimum: known optimum value for the objective function. Default is 0.
        - maxFES: maximum number of fitness evaluations.
        If set to None, will be calculated as 10000 * [number of dimensions] = 10000 * len(bounds)"""

        # Attribute initialization

        # From arguments

        if( len(self.bounds[0]) != len(self.bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        self.func = func
        self.bounds = bounds
        self.dimensions = len(self.bounds[0])
        self.crit = crit
        self.eliteSize = eliteSize
        self.mu = mu
        self.lamb = lamb
        self.optimum = optimum
        self.tol = tol
        self.equalWeights = equalWeights
        self.cc = self.cc
        self.ccov = self.ccov
        self.csigma = self.csigma
        self.dsigma = self.dsigma

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * len(bounds[0]) # 10000 x [dimensions]

        if(self.lamb == None):
            self.lamb = 4 + np.floor(3 * np.log(self.dimensions)) # note: np.log = ln

        if(self.mu == None):
            self.mu = np.floor(0.5 * self.lamb)

        self.weights = []

        if(self.equalWeights):
            self.weights = np.ones(self.mu)

        else:
            self.weights = np.log( (self.lamb + 1) / 2 ) - np.array([ np.log(i + 1) for i in range(self.mu) ]) # i+1: python is 0-indexed

        # Default CMA-ES parameter values
        self.cc = 4 / (self.dimensions + 4) # weight of covariance path for adaptation; its inverse indicates its duration in the algorithm
        self.ccov = 2 / np.power( ( self.dimensions + np.sqrt(2) ), 2) # weight of present covariance matrics for adaptation; its inverse indicates its duration in the algorithm
        self.csigma = 4 / (self.dimensions + 4) # weight of present sigma/stepsize for adaptation; its inverse indicates a sigma's duration in the algorithm
        self.dsigma = (1 / self.csigma) + 1 # damping constant for sigma adaptations

        self.B = np.eye(self.dimensions)
        self.D = np.eye(self.dimensions)
        self.BD = B * D
        self.C = self.BD * np.transpose(self.BD)
        self.cw = np.sum(self.weights) / np.linalg.norm(self.weights)
        self.chiN = np.sqrt(self.dimensions) * ( 1 - (1 / 4 * self.dimensions) + ( 1 / 21 * np.power(self.dimensions, 2) ) )

        # Control attributes
        self.elite = None # used in elitism
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.bestSoFar = 0
        self.results = None

        # start values
        self.xmeanw = np.ones(self.dimensions) # weighted mean of best solutions
        self.sigma = 1.0 # step size
        self.minSigma = 1e-15 # min step size
        self.fitness = np.array( [ (np.inf if self.crit == "min" else -np.inf) for i in range(self.lamb) ] )

    def execute(self):

        # metrics = self.getFitnessMetrics() # post-initialization: generation 0
        #
        # # Arrays for collecting metrics
        #
        # generations = [ self.genCount ]
        # FESCount = [ self.FES ]
        # errors = [ metrics["error"] ]
        # maxFits = [ metrics["top"] ]
        # maxPoints = [ metrics["topPoints"] ]
        # minFits = [ metrics["bottom"] ]
        # minPoints = [ metrics["bottomPoints"] ]
        # avgFits = [ metrics["avg"] ]

        try:

            while ( abs(self.bestSoFar - self.optimum) > self.tol ):

                # generating offspring
                X = np.zeros((self.dimensions, self.lamb)) # offspring
                Z = np.zeros((self.dimensions, self.lamb)) # random vectors, normally distributed
                # np.sum(axis=0)

                for k in range(self.lamb):

                    valid = False
                    z = None
                    new_x = None

                    while not valid:

                        # column vectors
                        z = np.random.normal(0, 1, (self.dims, 1))
                        new_x = self.xmeanw + self.sigma * (self.BD * z)

                        # bound checking
                        brokenBound = False
                        for l in range(self.dimensions):

                            if(new_x[l] < self.bounds[0][l] or new_x[l] > self.bounds[1][l]):
                                brokenBound = True
                                break

                        if(brokenBound): continue # repeats if bounds are broken

                        valid = True

                    # columns
                    X[:, k] = new_x
                    Z[:, k] = z

                    # evaluating function value
                    self.fitness[k] = self.func(np.transpose(new_x))
                    self.FES += 1
                    if self.FES == self.maxFES: raise MaxFESReached

                    # continue from here



                self.getElite() # gets the best values if self.eliteSize > 0; does nothing otherwise

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

                print(metrics["error"])

        except KeyboardInterrupt:
            return

    def calculateFitnessPop(self):
        """Calculates the fitness values for the entire population."""

        for ind in self.pop:
            ind[1] = self.func(np.array(ind[0][:self.dimensions]))
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

    def adaptiveCreepMutation(self, index, prob=1):
        """Executes a creep mutation on the individual (child) with a specified index."""

        # adds a random value to the gene with a probability prob

        newGenes = []
        fixedNormalMod = np.random.normal(0, 1)
        fixedTau = 1/np.sqrt(2 * self.dimensions)
        dimTau = 1/np.sqrt( 2 * np.sqrt(self.dimensions) )

        for i in range(self.dimensions):

            #modifying this dimension's sigma
            self.children[index][0][self.dimensions + i] *= np.exp(fixedTau * fixedNormalMod + dimTau * np.random.normal(0, 1))
            if(self.children[index][0][self.dimensions + i] < self.adaptiveEpsilon): self.children[index][0][self.dimensions + i] = self.adaptiveEpsilon

            #applying it to this dimension's parameter
            self.children[index][0][i] = self.children[index][0][i] + np.random.normal(0, self.children[index][0][self.dimensions + i]) if (np.random.uniform(0, 1) < prob) else self.children[index][0][i]

            if(self.children[index][0][i] < self.bounds[0][i]): self.children[index][0][i] = self.bounds[0][i]
            if(self.children[index][0][i] > self.bounds[1][i]): self.children[index][0][i] = self.bounds[1][i]
            #truncating to bounds

        self.children[index][1] = self.func(np.array(self.children[index][0][:self.dimensions]))
        self.FES += 1
        if self.FES == self.maxFES: raise MaxFESReached


    def isInBounds(self, ind):
        """Bound checking function for the genes. Used for mutation and crossover."""

        for i in range( self.dimensions ):

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

    def noParentSelection(self):
        self.matingPool = self.pop

    def tournamentSelection(self, crossover = True):
        # use with matingPool for parent selection
        # use with pop for post-crossover selection (non-generational selection schemes)

        winners = []

        if(not crossover):
            self.pop.extend(self.children)

            if(self.elite): # if there is an elite and it is not a crossover selection...
                for ind in self.elite:
                    winners.extend(self.elite)

        limit = self.popSize

        if(crossover):
            limit = self.matingPoolSize

        while len(winners) < limit:

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

        for i in range(self.numChildren):

            positions = np.random.randint(0, len(self.matingPool), 2)
            parent1, parent2 = self.matingPool[positions[0]], self.matingPool[positions[1]]

            if (np.random.uniform(0, 1) < crossProb): # crossover executed with probability crossProb

                child = []
                genes = []

                for j in range( len(parent1[0]) ): # iterate through its genes

                    while True:
                        beta = ( np.random.uniform( -alpha, 1 + alpha ) )
                        gene = parent1[0][j] + beta * (parent2[0][j] - parent1[0][j])

                        if (j < self.dimensions):

                            if( self.bounds[0][j] <= gene <= self.bounds[1][j] ):
                                genes.append(gene)
                                break
                                #Fora dos limites? Refazer.

                        else:

                            if(gene < self.adaptiveEpsilon):
                                gene = self.adaptiveEpsilon

                            genes.append(gene)

                            break

                child.append(genes)
                child.append(self.func(np.array(genes[:self.dimensions])))
                self.FES += 1
                if self.FES == self.maxFES: raise MaxFESReached

                children.append(child)

            else: #if it is not executed, the parent with the best fitness is given as a child
                if self.crit == "min": children.append(parent1) if parent1[1] <= parent2[1] else children.append(parent2) # compara valores de f. escolhe o de menor aptidão
                else: children.append(parent1) if parent1[1] >= parent2[1] else children.append(parent2) # compara valores de f. escolhe o de menor aptidão

        self.children = children

    def discreteCrossover(self, crossProb=1):
        if not self.matingPool:
            raise ValueError("There is no mating pool. Execute a selection function for it first.")

        children = []

        for i in range(self.numChildren):

            positions = np.random.randint(0, len(self.matingPool), 2)
            parents = [ self.matingPool[positions[0]], self.matingPool[positions[1]] ]

            if (np.random.uniform(0, 1) < crossProb): # crossover executed with probability crossProb

                child = []
                genes = []

                for j in range( len(parents[0][0]) ): # iterate through its genes

                    while True:

                        if (j < self.dimensions):

                            gene = parents[np.random.randint(0, 2)][0][j]

                            if( self.bounds[0][j] <= gene <= self.bounds[1][j] ):
                                genes.append(gene)
                                break
                                #Fora dos limites? Refazer.

                        else:

                            gene = (parents[0][0][j] + parents[1][0][j])/2 # indexes: parent, genotype, gene

                            if(gene < self.adaptiveEpsilon):
                                gene = self.adaptiveEpsilon

                            genes.append(gene)

                            break

                child.append(genes)
                child.append(self.func(np.array(genes[:self.dimensions])))
                self.FES += 1
                if self.FES == self.maxFES: raise MaxFESReached

                children.append(child)

            else: #if it is not executed, the parent with the best fitness is given as a child
                if self.crit == "min": children.append(parents[0]) if parents[0][1] <= parents[1][1] else children.append(parents[1]) # compara valores de f. escolhe o de menor aptidão
                else: children.append(parents[0]) if parents[0][1] >= parents[1][1] else children.append(parents[1]) # compara valores de f. escolhe o de menor aptidão

        self.children = children

    def intermediateCrossover(self, crossProb=1):
        if not self.matingPool:
            raise ValueError("There is no mating pool. Execute a selection function for it first.")

        children = []

        for i in range(self.numChildren):

            positions = np.random.randint(0, len(self.matingPool), 2)
            parents = [ self.matingPool[positions[0]], self.matingPool[positions[1]] ]

            if (np.random.uniform(0, 1) < crossProb): # crossover executed with probability crossProb

                child = []
                genes = []

                for j in range( len(parents[0][0]) ): # iterate through its genes

                    while True:

                        if (j < self.dimensions):

                            gene = (parents[0][0][j] + parents[1][0][j])/2 # indexes: parent, genotype, gene

                            if( self.bounds[0][j] <= gene <= self.bounds[1][j] ):
                                genes.append(gene)
                                break
                                #Fora dos limites? Refazer.

                        else:

                            gene = (parents[0][0][j] + parents[1][0][j])/2 # indexes: parent, genotype, gene

                            if(gene < self.adaptiveEpsilon):
                                gene = self.adaptiveEpsilon

                            genes.append(gene)

                            break

                child.append(genes)
                child.append(self.func(np.array(genes[:self.dimensions])))
                self.FES += 1
                if self.FES == self.maxFES: raise MaxFESReached

                children.append(child)

            else: #if it is not executed, the parent with the best fitness is given as a child
                if self.crit == "min": children.append(parents[0]) if parents[0][1] <= parents[1][1] else children.append(parents[1]) # compara valores de f. escolhe o de menor aptidão
                else: children.append(parents[0]) if parents[0][1] >= parents[1][1] else children.append(parents[1]) # compara valores de f. escolhe o de menor aptidão

        self.children = children

    def noCrossover(self):
        self.children = self.matingPool

    def generationalSelection(self):

        if self.crit == "max":
            self.children.sort(key = getSecond, reverse = True) # ordena pelo valor de f em ordem decrescente

        else:
            self.children.sort(key = getSecond, reverse = False) # ordena pelo valor de f em ordem crescente

        if self.eliteSize > 0:

            newPop = []
            newPop.extend(self.elite)
            newPop.extend(self.children)
            self.pop = newPop[:self.popSize] # cutting out the worst individuals

        else:

            if(self.numChildren < self.popSize):
                raise ValueError("There are fewer children than the population size. Please raise the number of children, in order to keep a constant population size.")
            newPop = []
            newPop.extend(self.children)
            self.pop = newPop[:self.popSize]

    def genitor(self):
        #excludes the worst individuals
        self.pop.extend(self.children)
        newPop = []

        if(self.elite):
            newPop.extend(self.elite)

        if self.crit == "max":
            self.pop.sort(key = getSecond, reverse = True) # ordena pelo valor de f em ordem decrescente

        else:
            self.pop.sort(key = getSecond, reverse = False) # ordena pelo valor de f em ordem crescente

        newPop.extend(self.pop)
        self.pop = newPop[:self.popSize] # cuts the worst individuals here

if __name__ == '__main__':

    # Test of the GA's performance over CEC2005's F1 (shifted sphere)

    import time
    from optproblems import cec2005

    bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    # AGA = AdaptiveGA(cec2005.F1(10), bounds, crit="min", optimum=-450, tol=1e-08, eliteSize=0, matingPoolSize=70, popSize=70, adaptiveEpsilon=1e-05)
    AGA = AdaptiveGA(cec2005.F4(10), bounds, crit="min", optimum=-450, tol=1e-08, eliteSize=1, numChildren=150, matingPoolSize=30, popSize=30, adaptiveEpsilon=1e-05)

    AGA.setParentSelection(AGA.noParentSelection, None )
    # AGA.setParentSelection(AGA.tournamentSelection, (True,) )
    # AGA.setCrossover(AGA.blxAlphaCrossover, (0.5, 1)) # alpha, prob
    AGA.setCrossover(AGA.discreteCrossover, (1,)) # prob
    # AGA.setCrossover(AGA.intermediateCrossover, (1,)) # prob
    # AGA.setCrossover(AGA.noCrossover, None)
    AGA.setMutation(AGA.adaptiveCreepMutation, (1,)) # prob
    # AGA.setNewPopSelection(AGA.genitor, None)
    AGA.setNewPopSelection(AGA.generationalSelection, None)
    # AGA.setNewPopSelection(AGA.tournamentSelection, None)
    AGA.execute()
    results = AGA.results

    print("AGA: for criterion = " + AGA.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    " (error of " + str(results["errors"][-1]) + ") (points " + str(results["minPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))


    # import sys
    # sys.path.append("../../cec2014/python") # Fedora
    # # sys.path.append("/mnt/c/Users/Lucas/Documents/git/cec2014/python") # Windows
    # import cec2014
    #
    # def func(arr):
    #     return cec2014.cec14(arr, 1)
    #
    # bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ] # 10-dimensional sphere (optimum: 0)
    #
    # start = time.time()
    #
    # # Initialization
    # AGA = AdaptiveGA(func, bounds, crit="min", optimum=100, tol=1e-08, eliteSize=0, numChildren=200, matingPoolSize=50, popSize=100, adaptiveEpsilon=1e-05)
    #
    # AGA.setParentSelection(AGA.tournamentSelection, (True,) )
    # AGA.setCrossover(AGA.blxAlphaCrossover, (0.5, 1)) # alpha, prob
    # AGA.setMutation(AGA.adaptiveCreepMutation, (1,)) # prob
    # AGA.setNewPopSelection(AGA.genitor, None)
    # AGA.execute()
    # results = AGA.results
    #
    # print("AGA: for criterion = " + AGA.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    # " (error of " + str(results["errors"][-1]) + ") (points " + str(results["minPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    # " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )
    #
    # end = time.time()
    # print("time:" + str(end - start))
