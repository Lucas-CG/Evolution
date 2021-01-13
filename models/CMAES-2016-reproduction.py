import numpy as np
from time import sleep
import sys

def getFirst(ind):
    return ind[0]

class MaxFESReached(Exception):
    """Exception used to interrupt the GA operation when the maximum number of fitness evaluations is reached."""
    pass

class CMAES(object):
    """Implements a real-valued CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, func, bounds, crit="min", equalWeights=False, optimum=0, maxFES=None, tol=1e-08):
        """Initializes the algorithm. Arguments:
        - func: a function name (the optimization problem to be resolved).
        - bounds: 2D array. bounds[0] has lower bounds; bounds[1] has upper bounds. They also define the size of individuals.
        - crit: criterion ("min" or "max").
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
        self.optimum = optimum
        self.tol = tol
        self.equalWeights = equalWeights

        if(maxFES): self.maxFES = maxFES
        else: self.maxFES = 10000 * len(bounds[0]) # 10000 x [dimensions]

        self.lamb = 4 + int(3 * np.log(self.dimensions)) # note: np.log = ln

        self.mu = int(self.lamb/2)

        self.weights = []
        self.weightSum = 0
        self.posWeightSum = 0
        self.negWeightSum = 0

        self.weights = np.log( (self.lamb + 1) / 2 ) - np.array([ np.log(i + 1) for i in range(int(self.lamb) ) ]) # i+1: python is 0-indexed
        self.mueff = np.power(np.sum(self.weights[:self.mu]), 2) / sum(np.power(w, 2) for w in self.weights[:self.mu]) # variance-effectiveness
        self.negMueff = np.power(np.sum(self.weights[self.mu:]), 2) / sum(np.power(w, 2) for w in self.weights[self.mu:]) # complement
        self.c1 = 2 / ( np.power((self.dimensions + 1.3), 2) + self.mueff ) # learning rate for rank-one update of C
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dimensions + 2)**2 + self.mueff)]) # and for rank-mu update
        # weights can be negative too (active CMA; prevents complex numbers)
        self.posWeightSum = np.sum(self.weights[:self.mu])
        self.negWeightSum = abs(np.sum(self.weights[self.mu:])) # abs value
        self.weightSum = self.posWeightSum + self.negWeightSum
        negAlphaMu = 1 + (self.c1/self.cmu)
        negAlphaMuEff = 1 + ( (2 * self.negMueff) / (self.mueff + 2) )
        negAlphaPosDef = (1 - self.c1 - self.cmu) / (self.dimensions * self.cmu)
        self.weights = np.array( [self.weights[i]/self.posWeightSum if i < self.mu else self.weights[i] * min([negAlphaMu, negAlphaMuEff, negAlphaPosDef]) / self.negWeightSum for i in range(self.lamb)] )
        # positive weights converge to 1

        # Default CMA-ES parameter values
        self.cc = (4 + self.mueff/self.dimensions) / (self.dimensions + 4 + 2 * self.mueff / self.dimensions) # weight of covariance path for adaptation; its inverse indicates its duration in the algorithm
        self.cs = (self.mueff + 2) / (self.dimensions + self.mueff + 5) # weight of present sigma/stepsize for adaptation; its inverse indicates a sigma's duration in the algorithm
        # self.ds = 2 * self.mueff/self.lamb + 0.3 + self.cs # damping constant for sigma adaptations # code
        self.ds = 1 + 2 * max( [ 0, np.sqrt( (self.mueff - 1) / (self.dimensions + 1) ) - 1 ] ) + self.cs # damping constant for sigma adaptations # paper
        self.ccov = 2 / np.power( ( self.dimensions + np.sqrt(2) ), 2) # weight of present covariance matrics for adaptation; its inverse indicates its duration in the algorithm

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
        self.weightedy = np.zeros(self.dimensions)
        self.zmeanw = np.zeros(self.dimensions)
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
                self.Y = np.zeros((self.dimensions, int(self.lamb)))
                self.Z = np.zeros((self.dimensions, int(self.lamb))) # random vectors, normally distributed

                try:

                    # generating offspring
                    for k in range(int(self.lamb)):

                        valid = False
                        z = None
                        y = None
                        new_x = None

                        while not valid:

                            # column vectors
                            z = np.random.normal(0, 1, (self.dimensions, 1))
                            y = self.sigma * ( np.matmul(self.BD, z) )
                            new_x = self.xmeanw + np.transpose(y)[0]

                            # bound checking
                            if not self.isInBounds(new_x): continue # repeats if bounds are broken

                            valid = True

                        # columns
                        self.X[:, k] = new_x
                        self.Y[:, k] = np.transpose(y)[0]
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

                xold = self.xmeanw

                print(self.fitness)

                # calculating weighted means
                self.xmeanw = sum( [ self.X[:, self.rankingIndexes[i]] * self.weights[i] for i in range(int(self.mu)) ] )
                self.zmeanw = sum( [ self.Z[:, self.rankingIndexes[i]] * self.weights[i] for i in range(int(self.mu)) ] )

                yilamb = np.zeros((self.dimensions, self.lamb))

                for i in range(self.lamb): yilamb[:, i] = (self.X[:, i] - self.xmeanw) / self.sigma
                self.yilambmean = sum( [ yilamb[:, self.rankingIndexes[i]] * self.weights[i] for i in range(int(self.mu)) ] )

                y = self.xmeanw - xold
                z = np.matmul( np.matmul( self.B, np.matmul(np.linalg.inv(self.D), np.transpose(self.B)) ), y) # C^(-1/2) (= B * D^(-1) * B^T) * y
                csn = np.sqrt( self.cs * (2 - self.cs) * self.mueff ) / self.sigma

                # sigma path
                self.ps = (1 - self.cs) * self.ps + csn * z

                # adapting sigma
                self.sigma = self.sigma * np.exp( (self.cs/self.ds) * ( ( (np.linalg.norm(self.ps)) / self.chiN ) - 1 ) )

                hs = 1 if ( (np.linalg.norm(self.ps)) / np.sqrt(1 - (1-self.cs)**(2 * (self.genCount + 1) ) ) ) < ( 1.4 + (2 / ( self.dimensions + 1 ) ) ) * self.chiN else 0

                # adapting covariance matrix
                self.pc = (1 - self.cc) * self.pc + hs * ( np.sqrt( self.cc * (2 - self.cc) * self.mueff ) ) * self.yilambmean[:, None]
                covWeights = np.array([ self.weights[i] if i < self.mu else self.dimensions / np.linalg.norm(np.matmul( np.matmul( self.B, np.matmul(np.linalg.inv(self.D), np.transpose(self.B) ) ), yilamb[:, i] ) ) ** 2 for i in range(self.lamb) ])
                # transpose: xmeanw and zmeanw are line vectors
                deltaHs = (1 - hs) * self.cc * (2 - self.cc)
                self.C = (1 + self.c1 * deltaHs - self.c1 - self.cmu * self.weightSum) * self.C + \
                         self.c1 * ( self.pc * np.transpose(self.pc[0]) ) + \
                         self.cmu * sum( [ covWeights[i] * yilamb[:, i] * yilamb[:, i][:, None] for i in range(self.lamb) ] )

                # [:, None] transposes a 1D array

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
