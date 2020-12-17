from optproblems import cec2005
from os import makedirs
import statistics
import csv
import argparse
import numpy as np
import sys
sys.path.append("../cec2014/python") # Fedora
# sys.path.append("/mnt/c/Users/Lucas/Documents/git/cec2014/python") # Windows
import cec2014

parser = argparse.ArgumentParser(description="Run experiments with an algorithm specified in the input.")
parser.add_argument("--algorithm", dest='algorithm', help="Name of the algorithm (can be ACO, ABC, DE, GA, AGA, PSO, RegPSO or SSO).")
algorithm = parser.parse_args(sys.argv[1:]).algorithm.upper()
print(algorithm)

if algorithm == "ACO":
    from models import AntColonyOptimization

if algorithm == "ABC":
    from models import ArtificialBeeColony

if algorithm == "DE":
    from models import DifferentialEvolution

if algorithm == "GA":
    from models import GeneticAlgorithm

if algorithm == "AGA":
    from models import AdaptiveGA

if algorithm == "PSO":
    from models import ParticleSwarmOptimization

if algorithm == "REGPSO":
    from models import RegPSO

if algorithm == "SSO":
    from models import SocialSpiderOptimization

def F1(arr):
    return cec2014.cec14(np.array(arr), 1)

def F2(arr):
    return cec2014.cec14(np.array(arr), 2)

def F4(arr):
    return cec2014.cec14(np.array(arr), 4)

def F6(arr):
    return cec2014.cec14(np.array(arr), 6)

def F7(arr):
    return cec2014.cec14(np.array(arr), 7)

def F9(arr):
    return cec2014.cec14(np.array(arr), 9)

def F14(arr):
    return cec2014.cec14(np.array(arr), 14)

if __name__ == '__main__':

    dims = 10
    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ]
    functions = [F1, F2, F4, F6, F7, F9, F14]
    funIndexes = [1, 2, 4, 6, 7, 9, 14]
    optimums = [100, 200, 400, 600, 700, 900, 1400]
    FESThresholds = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    numRuns = 25

    # Creating result path
    pathName = "Results/" + algorithm
    makedirs(pathName, exist_ok=True)

    # Initializing result files

    tableFileName = pathName + "/" + algorithm + "_" + str(dims) + "D.csv"

    with open(tableFileName, "w") as resultsFile:

        writer = csv.writer(resultsFile, delimiter = ',')

        writer.writerow( ["Fi_10D", "Best", "Worst", "Median", "Mean", "Std_Dev", "Success_Rate"] )


    for i in range(len(functions)):

        plotFileName = pathName + "/" + algorithm + "_F" + str(funIndexes[i]) + "_" + str(dims) + "D_Plot.csv"

        with open(plotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')

            header = ["FES_Multiplier"]
            header.extend( ["Run " + str(i+1) for i in range(numRuns)] )
            header.append("Average")

            writer.writerow( header )

        bestPlotFileName = pathName + "/" + algorithm + "_F" + str(funIndexes[i]) + "_" + str(dims) + "D_BestPlot.csv"

        with open(bestPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )

        worstPlotFileName = pathName + "/" + algorithm + "_F" + str(funIndexes[i]) + "_" + str(dims) + "D_WorstPlot.csv"

        with open(worstPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )

        errors = []
        successes = []
        FESResults = []

        for j in range(numRuns):

            results = None

            if algorithm == "ACO":
                model = AntColonyOptimization(functions[i], bounds, numAnts=2, optimum=optimums[i])
                model.execute()
                results = model.results

            if algorithm == "ABC":
                model = ArtificialBeeColony(functions[i], bounds, popSize=50, workerOnlookerSplit=0.5, limit=None, numScouts=1, optimum=optimums[i])
                model.execute()
                results = model.results

            if algorithm == "DE":
                model = DifferentialEvolution(functions[i], bounds, optimum = optimums[i])
                model.setMutation(model.classicMutation, ("rand", 0.5, 1)) # base, F, nDiffs
                model.setCrossover(model.classicCrossover, ("bin", 0.5)) # type, CR
                model.execute()
                results = model.results

            if algorithm == "GA":
                model = GeneticAlgorithm(functions[i], bounds, crit="min", optimum=optimums[i], tol=1e-08, eliteSize=0, matingPoolSize=100, popSize=100)
                model.setParentSelection(model.tournamentSelection, (True,) )
                model.setCrossover(model.blxAlphaCrossover, (0.5, 1)) # alpha, prob
                model.setMutation(model.uniformMutation, (0.05, )) # prob, mean, sigma
                model.setNewPopSelection(model.genitor, None)
                model.execute()
                results = model.results

            if algorithm == "AGA":
                model = AdaptiveGA(functions[i], bounds, crit="min", optimum=optimums[i], tol=1e-08, eliteSize=0, matingPoolSize=70, popSize=70, adaptiveEpsilon=1e-05)
                model.setParentSelection(model.tournamentSelection, (True,) )
                model.setCrossover(model.blxAlphaCrossover, (0.5, 1)) # alpha, prob
                model.setMutation(model.adaptiveCreepMutation, (1,)) # prob
                model.setNewPopSelection(model.genitor, None)
                model.execute()
                results = model.results

            if algorithm == "PSO":
                model = ParticleSwarmOptimization(functions[i], bounds, popSize=80, globalWeight=2.05, localWeight=2.05, clerkK=False, inertiaDecay=True, optimum=optimums[i])
                model.execute()
                results = model.results

            if algorithm == "REGPSO":
                model = RegPSO(functions[i], bounds, popSize=80, clerkK=False, inertiaDecay=True, optimum=optimums[i], prematureThreshold=1.1e-06)
                model.execute()
                results = model.results

            if algorithm == "SSO":
                model = SocialSpiderOptimization(functions[i], bounds, popSize=30, PF=0.7, normalizeDistances=True, optimum=optimums[i])
                model.execute()
                results = model.results

            # Treating results

            errors.append(results["errors"][-1])
            successes.append( int(results["errors"][-1] <= model.tol) )
            # True/False values of successes are converted to int, to fetch the mean (success rate)

            # getting errors for different FES values

            currentFES = 0
            FESThresholdErrors = [] # errors for each FES multiplier

            for k in range(len(FESThresholds)):

                l = 0
                endReached = False

                for m in range ( l, len( results["FESCounts"] ) ):

                    currentFES = results["FESCounts"][m]

                    # FES value coincides with the threshold or has surpassed it
                    if(currentFES >= FESThresholds[k] * model.maxFES):
                        FESThresholdErrors.append(results["errors"][m])
                        l = k + 1
                        break

                    elif ( m == len( results["FESCounts"] ) - 1 ):
                        FESThresholdErrors.append(results["errors"][m])
                        l = k + 1
                        endReached = True
                        break

                if(endReached): break


            FESResults.append(FESThresholdErrors) # each line is an execution

        # End of the 25 runs. Write the results to the table and plot files.

        # Result tables

        with open(tableFileName, "a") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')

            writer.writerow( [str(funIndexes[i]), min(errors), max(errors), statistics.median(errors),
            statistics.mean(errors), statistics.stdev(errors), statistics.mean(successes)] )
            # statistics.mean(successes) works, even if it is a list of bools

        # Normal plots
        with open(plotFileName, "a") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            lastValidVals = [0 for j in range(numRuns)]

            for i in range(len(FESThresholds)):

                row = [FESThresholds[i]]
                rowVals = []

                for j in range(numRuns):

                    if( i < len(FESResults[j]) ):
                        row.append(FESResults[j][i])
                        rowVals.append(FESResults[j][i])
                        lastValidVals[j] = FESResults[j][i]

                    else:
                        row.append("None")
                        rowVals.append(lastValidVals[j]) # for the mean, use the last valid error value

                mn = statistics.mean(rowVals)
                row.append(mn)
                writer.writerow(row)

        # Finding best run
        bestRunId = errors.index(min(errors))

        # Best run plot
        with open(bestPlotFileName, "a") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')

            for i in range(len(FESThresholds)):

                if( i >= len(FESResults[bestRunId]) ): break
                row = [FESThresholds[i], FESResults[bestRunId][i]]
                writer.writerow(row)

        # Finding worst run
        worstRunId = errors.index(max(errors))

        # Worst run plot
        with open(worstPlotFileName, "a") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')

            for i in range(len(FESThresholds)):

                if( i >= len(FESResults[worstRunId]) ): break
                row = [FESThresholds[i], FESResults[worstRunId][i]]
                writer.writerow(row)