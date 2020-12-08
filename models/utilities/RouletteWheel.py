import numpy as np

class RouletteWheel(object):
    """Defines a roulette wheel selection method. For initialization, it requires an array of weights,
    from which probabilities are extracted."""

    def __init__(self, weights):

        self.weights = weights

        self.numSlots = len(weights)

        weightSum = sum(weights)

        probabilities = weights / weightSum

        self.slots = [] # stores cumulative probabilities

        cumProbSum = 0

        for i in range(self.numSlots):
            cumProbSum += probabilities[i]
            self.slots.append(cumProbSum) # 0.1, 0.2, ..., 1.0

    def draw(self):

        draw = np.random.uniform(0, 1)

        for j in range(self.numSlots):

            if draw < self.slots[j]:
                return j
