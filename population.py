import chromossome
import numpy as np

class Population(object):
    """Implements a population.
    Attributes: size (number of individuals), func (fitness function)"""

    def __init__(self, size, func):
        self.size = size
        self.func = func
        self.pop = None

    def createRandom():
        self.pop = np.array( [ Chromossome(size, func, lowerBounds, upperBounds, True, None) for i in range(size) ] )

    def calculateFitness():

        for ind in self.pop:
            ind.calculateFitness()

    def getHighest():
        # Finds the individuals with the highest fitness value of the population.
        # Returns (top, points) -> top = fitness value / points: list of the individuals' genes

        top = -np.inf
        points = []

        for ind in self.pop:

            if (top < ind.fval):
                top = ind.fval
                points = [ ind.genes ]

            elif (top == ind.fval):
                points.append(ind.genes)

        return (top, points)

    def getLowest():

        # Finds the individuals with the lowest fitness value of the population.
        # Returns (bottom, points) -> bottom = fitness value / points: list of the individuals' genes

        bottom = np.inf
        points = []

        for ind in self.pop:

            if (bottom > ind.fval):
                bottom = ind.fval
                points = [ ind.genes ]

            elif (bottom == ind.fval):
                points.append(ind.genes)

        return (bottom, points)

    def getMean():

        # Finds the mean fitness value of the population.
        # Returns avg -> mean fitness value

        total = 0

        for ind in self.pop:

            total += ind.fval

        return total/self.size

    def getFitnessMetrics():

        # Finds the mean, greater and lower fitness values for the population,
        # as well as the points with the greater and lower ones.
        # Returns a dict, whose keys are:
        # "avg" to average value
        # "top" to top value
        # "topPoints" to a list of points with the top value
        # "bottom" to bottom value
        # "bottomPoints" to a list of points with the bottom value

        # This function is intended to save iterations through the population, when
        # more than one iterative operation is necessary.

        total = 0
        top = -np.inf
        topPoints = []
        bottom = np.inf
        bottomPoints = []

        for ind in self.pop:

            total += ind.fval

            if (top < ind.fval):
                top = ind.fval
                topPoints = [ ind.genes ]

            elif (top == ind.fval):
                topPoints.append(ind.genes)

            if (bottom > ind.fval):
                bottom = ind.fval
                bottomPoints = [ ind.genes ]

            elif (bottom == ind.fval):
                bottomPoints.append(ind.genes)

        avg = total/self.size

        return {"avg": avg, "top": top, "topPoints": topPoints, "bottom": bottom, bottomPoints: "bottomPoints"}

    def createUniform():
        pass
        #To do: create uniformly distributed elements
