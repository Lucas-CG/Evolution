from GeneticAlgorithm import GeneticAlgorithm
from optproblems import cec2005
import time

if __name__ == '__main__':

    bounds = [ [-100 for i in range(10)], [100 for i in range(10)] ]
    # 10 dimensions; each dimension variable varies within [-100, +100]

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
    " and " + str(results["FESCount"][-1]) + " fitness evaluations" )

    end = time.time()
    print("time:" + str(end - start))
