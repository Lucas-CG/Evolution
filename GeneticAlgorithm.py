import numpy as np

def getSecond(ind):
    return ind[1]

class GeneticAlgorithm(object):
    """Implements a real-valued Genetic Algorithm."""

    def __init__(self, func, bounds, popSize=100, randomPop=True, crit="min", eliteSize=0):
        """Initializes the population. Arguments:
        - func: a function name (the optimization problem to be resolved)
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - popSize: population size
        - randomPop: initialize the population as random (uniform)
        - crit: criterion ("min" or "max")
        - eliteSize: integer; defines whether elitism is enabled or not"""

        self.func = func
        self.bounds = bounds
        self.popSize = popSize
        self.randomPop = randomPop
        self.crit = crit
        self.eliteSize = eliteSize

        # Internal attributes

        self.pop = None

        if( len(self.bounds[0]) != len(self.bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        if self.randomPop:
            self.pop = [ [np.random.uniform(self.bounds[0], self.bounds[1]).tolist(), 0] for i in range(popSize) ] # genes, fitness
            # tolist(): convert to python list

        self.calculatedFits = False
        self.matingPool = None # used for parent selection
        self.children = None
        self.elite = None # used in elitism
        self.FES = 0 # function evaluations
        self.itcount = 0
        self.bestSoFar = 0
        self.mutation = None
        self.mutationParams = None
        self.parentSelection = None
        self.parentSelectionParams = None
        self.newPopSelection = None
        self.newPopSelectionParams = None

    def setMutation(self, mutation, mutationParams):
        """Configure the used mutation process. Parameters:
        - mutation: a mutation function
        - mutationParams: its parameters (a tuple)"""
        self.mutation = mutation
        self.mutationParams = mutationParams

    def setCrossover(self, crossover, crossoverParams):
        """Configure the used mutation process. Parameters:
        - crossover: a crossover function
        - crossoverParams: its parameters (a tuple)"""
        self.crossover = crossover
        self.crossoverParams = crossoverParams

    def setParentSelection(self, parentSelection, parentSelectionParams):
        """Configure the used parent selection process. Parameters:
        - parentSelection: a selection function
        - parentSelectionParams: its parameters (a tuple)"""
        self.parentSelection = parentSelection
        self.parentSelectionParams = parentSelectionParams

    def setNewPopSelection(self, newPopSelection, newPopSelectionParams):
        """Configure the used new population selection process. Parameters:
        - newPopSelection: a selection function
        - newPopSelectionParams: its parameters (a tuple)"""
        self.newPopSelection = newPopSelection
        self.newPopSelectionParams = newPopSelectionParams

    def calculateFitnessPop(self):
        """Calculates the fitness values for the entire population."""

        for ind in self.pop:
            ind[1] = self.func(*ind[0])
            self.FES += 1

        self.calculatedFits = True

    def run(self):
        #To do: define run
        pass

    def calculateFitnessInd(self, index):
        """Calculates the fitness values for a specific element (specified by index)."""

            self.pop[index][1] = self.func(*self.pop[index][0])
            self.FES += 1

    def getMax(self):
        """Finds the individuals with the highest fitness value of the population.
        Returns (top, points) -> top = fitness value / points: list of the individuals' genes"""

        if not self.calculatedFits:
            self.calculateFitnessPop()

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
        Returns (bottom, points) -> bottom = fitness value / points: list of the individuals' genes"""

        if not self.calculatedFits:
            self.calculateFitnessPop()

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
        """Returns the population's mean fitness value."""

        if not self.calculatedFits:
            self.calculateFitnessPop()

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

        This function is intended to save iterations through the population, when
        more than one iterative operation is necessary."""

        if not self.calculatedFits:
            self.calculateFitnessPop()

        total = 0
        top = -np.inf
        topPoints = []
        bottom = np.inf
        bottomPoints = []

        for i in range(self.popSize):

            total += self.self.pop[i][1]

            if (top < self.self.pop[i][1]):
                top = self.self.pop[i][1]
                topPoints = [ self.pop[i][0] ]

            elif (top == self.fits[i]):
                topPoints.append(self.pop[i][0])

            if (bottom > self.pop[i][1]):
                bottom = self.pop[i][1]
                bottomPoints = [ self.pop[i][0] ]

            elif (bottom == self.pop[i][1]):
                bottomPoints.append(self.pop[i][0])

        avg = total/self.popSize

        if(self.crit == "min"): self.bestSoFar = bottom
        if(self.crit == "max"): self.bestSoFar = top

        return {"avg": avg, "top": top, "topPoints": topPoints, "bottom": bottom, bottomPoints: "bottomPoints"}

    def creepMutation(self, prob=0.05, mean=0, stdev=1, calculateFitness=False):
        """Executes a creep mutation on the entire population. Use calculateFitness = True if you employ a non-generational
        selection procedure just after the mutation."""

        for i in range(self.popSize):

            while True:
                # adds a random value to the gene with a probability prob
                newGenes = [ gene + np.random.normal(mean, stdev) if (np.random.uniform(0, 1) < prob) else gene for gene in self.pop[i][0] ]

                #redo bound check
                if( self.isInBounds(self.pop[i]) ): self.pop[i][0] = newGenes

        if calculateFitness: self.calculateFitnessPop()

    def isInBounds(self, ind):
        # Bound checking function for the genes. Used for mutation and crossover.

        for i in range( len(ind) ):

            if not (self.bounds[0][i] <= ind[i] <= self.bounds[1][i]): return False
            # if this gene is in the bounds, inBounds keeps its True value.
            # else, it automatically returns False. Escaping to save up iterations.

        return True # if it has exited the loop, the genes are valid

    def getElite(self):

        if self.eliteSize > 0:

            if not self.calculatedFits:
                self.calculateFitness()

            elite = None

            if self.crit = "max":
                self.pop.sort(key = getSecond, reverse = True) # ordena pelo valor de f em ordem decrescente
                elite = self.pop[:self.eliteSize]

            else:
                self.pop.sort(key = getSecond, reverse = False) # ordena pelo valor de f em ordem crescente
                elite = self.pop[:self.eliteSize]

            self.elite = elite


    def tournamentSelection(self, crossover = True):
        # use with matingPool for parent selection
        # use with pop for post-crossover selection (non-generational selection schemes)
        # elite is for elitism in post-crossover selection. number of individuals

        winners = []

        if(not crossover and self.elite): # if there is an elite and it is not a crossover selection...
            for ind in elite:
                winners.extend(elite)

        while len(winners) < self.popSize:

            positions = np.random.randint(0, len(self.pop), 2)
            # len(self.pop) because the population may have children (larger than self.popSize)
            ind1, ind2 = pop[positions[0]], pop[positions[1]]
            if self.crit == "min": winner = ind1 if ind1[1] <= ind2[1] else ind2 # compara valores de f. escolhe o de menor aptidão
            else: winner = ind1 if ind1[1] >= ind2[1] else ind2 # compara valores de f. escolhe o de menor aptidão
            winners.append(winner)

        if crossover: self.matingPool = winners

        return winners

    def blxAlphaCrossover(self, alpha=0.5, crossProb=0.6):
        # Define o BLX-α crossover para toda a população (ver página 25 do pdf - capítulo 3)
        if not self.matingPool:
            raise ValueError("There is no mating pool. Execute a selection function for it first.")

        children = []

        for i in range(self.popSize):

            positions = np.random.randint(0, len(self.matingPool), 2)
            parent1, parent2 = self.matingPool[positions[0]], self.matingPool[positions[1]]
            child = []

            for j in range( len(parent1) ):

                while True:
                    beta = ( np.random.uniform( -alpha, 1 + alpha ) )
                    gene = parent1[j] + beta * (parent2[j] - parent1[j])

                    if( self.bounds[0][j] <= gene <= self.bounds[1][j] ):
                        child.append(gene)
                        #Fora dos limites? Refazer.

            children.append(child)

        self.children = children

        if(self.newPopSelection != self.generationalSelection): self.pop.extend(self.children)

    def generationalSelection():
        self.pop = self.children
