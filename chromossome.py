import numpy as np

class Chromossome(object):
    """Implements a real-valued chromossome for Evolutionary Computing algorithms.
    Attributes: size (number of genes), func (fitness function)
    and (lower and upper)Bounds (size "size" arrays indicating lower and upper bounds)"""

    def __init__(self, size, func, lowerBounds, upperBounds, random=True, prebuilt=None):
        # If random is True, initialize with random genes. If prebuilt is a set of genes,
        # the individual sets them and calculates the value of func

        self.genes = None
        self.size = size
        self.func = func
        self.fval = 0
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        if random: self.randomize()
        if prebuilt: self.set(prebuilt)

    def randomize():
        self.genes = np.random.uniform(self.lowerBounds, self.upperBounds)

    def set(genes):
        # Sets the individual to the specified genes.

        if(len(genes) == self.size):
            self.genes = genes

        else:
            raise ValueError( "Genes of invalid size. This chromossome is of size " + str(self.size) + ", but has received" + str( len(genes) ) + " genes." )

    def calculateFitness():
        self.fval = func(*self.genes)
        # the asterisk unpacks the gene array to use the elements as func's arguments

    def creepMutation(prob, mean=0, stdev=1):

        while True:
            # adds a random value to the gene with a probability prob
            newGenes = np.array( [ i + np.random.normal(mean, stdev) if (np.random.uniform(0, 1) < prob) else i for i in self.genes ] )

            #redo bound check
            if( isInBounds() ):
                return [ newGenes, f( newGenes[0], newGenes[1] ) ]
                # já calculo o novo valor de f após a mutação

    def isInBounds():
        # Bound checking function for the genes. Used for mutation and crossover.

        for i in range(len(self.genes)):

            if not (self.lowerBounds[i] <= self.genes[i] <= self.upperBounds[i]): return False
            # if this gene is in the bounds, inBounds keeps its True value.
            # else, it automatically returns False. Escaping to save up iterations.

        return True # if it has exited the loop, the genes are valid

    def getFitness():
        return self.fval
