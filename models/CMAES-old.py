import numpy as np
from time import sleep

def getFirst(ind):
    return ind[0]

class MaxFESReached(Exception):
    """Exception used to interrupt the GA operation when the maximum number of fitness evaluations is reached."""
    pass

class CMAES(object):
    """Implements a real-valued CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, func, bounds, crit="min", mu=None, lamb=None, equalWeights=False, cc=None, ccov=None, cs=None, ds=None, optimum=0, maxFES=None, tol=1e-08):
        """Initializes the algorithm. Arguments:
        - func: a function name (the optimization problem to be resolved).
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - crit: criterion ("min" or "max").
        - mu: parent number: the best individuals selected at each generation.
        - lamb: population size: generated individuals, which will be selected.
        - equalWeights: bool: defines whether "stronger" individuals will have greater weights at the crossover.
        - cc: present path's weight. Its inverse indicates this information's duration on the adaptation procedure.
        - ccov: present covariance matrix's weight. Its inverse indicates this information's duration on the adaptation procedure.
        - cs: present sigma/step size's weight. Its inverse indicates this information's duration on the adaptation procedure.
        - ds: damping constant for adaptation of step sizes.
        - optimum: known optimum value for the objective function. Default is 0.
        - maxFES: maximum number of fitness evaluations.
        If set to None, will be calculated as 10000 * [number of dimensions] = 10000 * len(bounds)"""

        # Attribute initialization

        # From arguments

        if( len(bounds[0]) != len(bounds[1]) ):
            raise ValueError("The bound arrays have different sizes.")

        self.func = func
        self.bounds = bounds
        self.dimensions = len(self.bounds[0])
        self.crit = crit
        self.mu = mu
        self.lamb = lamb
        self.optimum = optimum
        self.tol = tol
        self.equalWeights = equalWeights
        self.cc = cc
        self.ccov = ccov
        self.cs = cs
        self.ds = ds

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * len(bounds[0]) # 10000 x [dimensions]

        if(self.lamb == None): self.lamb = 4 + int(3 * np.log(self.dimensions)) # note: np.log = ln

        if(self.mu == None): self.mu = int(0.5 * self.lamb)

        self.weights = []

        if(self.equalWeights):
            self.weights = np.ones(int(self.mu))

        else:
            self.weights = np.log( (self.lamb + 1) / 2 ) - np.array([ np.log(i + 1) for i in range(int(self.mu) ) ]) # i+1: python is 0-indexed
            weightSum = np.sum(sum(self.weights))
            self.weights = self.weights/weightSum # guarateeing that the sum is 1

        # Default CMA-ES parameter values
        if self.cc == None: self.cc = 4 / (self.dimensions + 4) # weight of covariance path for adaptation; its inverse indicates its duration in the algorithm
        if self.ccov == None: self.ccov = 2 / np.power( ( self.dimensions + np.sqrt(2) ), 2) # weight of present covariance matrics for adaptation; its inverse indicates its duration in the algorithm
        if self.cs == None: self.cs = 4 / (self.dimensions + 4) # weight of present sigma/stepsize for adaptation; its inverse indicates a sigma's duration in the algorithm
        if self.ds == None: self.ds = (1 / self.cs) + 1 # damping constant for sigma adaptations

        # start values
        self.B = np.eye(self.dimensions) # cov matrix decomposition
        self.D = np.eye(self.dimensions) # cov matrix decomposition
        self.BD = np.matmul(self.B, self.D)
        self.C = np.matmul(self.BD, np.transpose(self.BD)) # cov matrix
        self.pc = np.zeros((self.dimensions, 1)) # path of cov matrix
        self.ps = np.zeros((self.dimensions, 1)) # path of sigma
        self.cw = np.sum(self.weights) / np.linalg.norm(self.weights)
        self.chiN = np.sqrt(self.dimensions) * ( 1.0 - (1.0 / 4 * self.dimensions) + ( 1.0 / 21 * np.power(self.dimensions, 2) ) )

        self.xmeanw = np.zeros(self.dimensions) # weighted mean of best solutions
        self.zmeanw = np.zeros(self.dimensions) # weighted mean of best solutions
        self.sigma = 60.0 # step size
        self.minSigma = 1e-15 # min step size
        self.fitness = np.array( [ (np.inf if self.crit == "min" else -np.inf) for i in range(int(self.lamb)) ] )
        self.rankingIndexes = None

        self.X = None
        self.Z = None

        # Control attributes
        self.elite = None # used in elitism
        self.FES = 0 # function evaluations
        self.genCount = 0
        self.bestSoFar = 0
        self.results = None

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

            generations = []
            FESCount = []
            errors = []
            maxFits = []
            maxPoints = []
            minFits = []
            minPoints = []
            avgFits = []

            while ( abs(self.bestSoFar - self.optimum) > self.tol ):

                # generating offspring
                self.X = np.zeros((self.dimensions, int(self.lamb))) # offspring
                self.Z = np.zeros((self.dimensions, int(self.lamb))) # random vectors, normally distributed

                try:

                    # generating offspring
                    for k in range(int(self.lamb)):

                        valid = False
                        z = None
                        new_x = None

                        while not valid:

                            # column vectors
                            z = np.random.normal(0, 1, (self.dimensions, 1))
                            new_x = self.xmeanw + self.sigma * ( np.transpose(np.matmul(self.BD, z))[0] )

                            # bound checking
                            if not self.isInBounds(new_x): continue # repeats if bounds are broken

                            valid = True

                        # columns
                        self.X[:, k] = new_x
                        self.Z[:, k] = np.transpose(z)[0]

                        # evaluating function value
                        self.fitness[k] = self.func(new_x)
                        self.FES += 1
                        if self.FES == self.maxFES: raise MaxFESReached

                except MaxFESReached:
                    break

                # sorting by fitness

                sortFitness = None

                if(self.crit == "min"):

                    sortFitness = [ [ self.fitness[i], i ] for i in range(int(self.lamb))]
                    sortFitness.sort(key=getFirst, reverse=False)

                else:

                    sortFitness = [ [ self.fitness[i], i ] for i in range(int(self.lamb))]
                    sortFitness.sort(key=getFirst, reverse=True)

                self.fitness = np.array([ sortFitness[i][0] for i in range(int(self.lamb)) ])
                self.rankingIndexes = np.array([ sortFitness[i][1] for i in range(int(self.lamb)) ])
                del sortFitness

                # calculating weighted means
                self.xmeanw = sum( [ self.X[:, self.rankingIndexes[i]] * self.weights[i] for i in range(int(self.mu)) ] ) / sum(self.weights)
                self.zmeanw = sum( [ self.Z[:, self.rankingIndexes[i]] * self.weights[i] for i in range(int(self.mu)) ] ) / sum(self.weights)

                # adapting covariance matrix

                self.pc = (1 - self.cc) * self.pc + ( ( np.sqrt( self.cc * (2 - self.cc) ) * self.cw ) * ( np.matmul( self.BD, np.transpose(self.zmeanw) ) ) )[:, None]
                # transpose: xmeanw and zmeanw are line vectors
                self.C = (1 - self.ccov) * self.C + self.ccov * ( self.pc * np.transpose(self.pc)[0] ) # [:, None] transposes an 1D array

                # adapting sigma
                self.ps = (1 - self.cs) * self.ps + ( np.sqrt( self.cs * (2 - self.cs) ) * self.cw ) * np.matmul(self.B, self.zmeanw)
                self.sigma = self.sigma * np.exp( (1/self.ds) * ( ( np.linalg.norm(self.ps) - self.chiN ) / self.chiN ) )

                # updating B and D from C
                if np.mod(self.FES/self.lamb, self.dimensions/10) < 1:

                    self.C = np.triu(self.C) + np.transpose( np.triu(self.C, 1) ) # enforce symmetry
                    self.D, self.B = np.linalg.eig(self.C)
                    # np.eig first return is return, by standard, a column containing the eigenvectors
                    self.D = np.diag(self.D) # converting the column to a diag. matrix

                    # limiting condition of C to 1e14 + 1
                    if ( np.amax( np.diag(self.D) ) ) > 1e14 * np.amin( np.diag(self.D) ):
                        tmp = np.amax(np.diag(self.D))/(1e14 - np.amin( np.diag(self.D) ) )
                        self.C = self.C + tmp * np.eye(self.dimensions)
                        self.D = self.D + tmp * np.eye(self.dimensions)

                    self.D = np.diag( np.sqrt( np.diag(self.D) ) ) # D contains standard deviations now
                    self.BD = np.matmul(self.B, self.D) # computing BD for speed up

                # adjusting minimal step size

                if (self.sigma * np.amin( np.diag(self.D) ) < self.minSigma) or self.fitness[0] == self.fitness[int(min(self.mu, self.lamb - 1))] or np.all(self.xmeanw == self.xmeanw + 0.2 * self.sigma * self.BD[:, int(np.floor( np.mod( self.FES/self.lamb, self.dimensions) ) ) ] ):
                    self.sigma = 1.4 * self.sigma

                # end generation
                # collecting metrics

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

                print( "%s, %s, %s"%( self.genCount, self.FES, metrics["error"] ) )

        except KeyboardInterrupt:
            return

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

        for i in range(int(self.lamb)):

            total += self.fitness[i]

            if (top < self.fitness[i]):
                top = self.fitness[i]
                topPoints = [ self.X[:, i] ]

            elif (top == self.fitness[i]):
                topPoints.append( self.X[:, i] )

            if (bottom > self.fitness[i]):
                bottom = self.fitness[i]
                bottomPoints = [ self.X[:, i] ]

            elif (bottom == self.fitness[i]):
                bottomPoints.append(self.X[:, i])

        topPoints = np.array(topPoints)
        bottomPoints = np.array(bottomPoints)

        avg = total/self.lamb

        if(self.crit == "min"): self.bestSoFar = bottom
        if(self.crit == "max"): self.bestSoFar = top

        error = abs(self.bestSoFar - self.optimum)

        return {"avg": avg, "top": top, "topPoints": topPoints, "bottom": bottom, "bottomPoints": bottomPoints, "error": error}

    def isInBounds(self, ind):
        """Bound checking function for the genes."""

        for i in range( self.dimensions ):

            if not (self.bounds[0][i] <= ind[i] <= self.bounds[1][i]): return False
            # if this gene is in the bounds, inBounds keeps its True value.
            # else, it automatically returns False. Escaping to save up iterations.

        return True # if it has exited the loop, the genes are valid


if __name__ == '__main__':

    # Test of the GA's performance over CEC2005's F3

    import time
    from optproblems import cec2005

    bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ] # 10-dimensional sphere (optimum: 0)

    start = time.time()

    # Initialization
    es = CMAES(cec2005.F3(10), bounds, crit="min", optimum=-450, tol=1e-08)
    es.execute()
    results = es.results

    print("CMA-ES: for criterion = " + es.crit + ", reached optimum of " + str(results["minFits"][-1]) +
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
