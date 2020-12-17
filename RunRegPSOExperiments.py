from models import RegPSO
from optproblems import cec2005
from os import makedirs
import statistics
import csv
import sys
import numpy as np
sys.path.append("../cec2014/python") # Fedora
# sys.path.append("/mnt/c/Users/Lucas/Documents/git/cec2014/python") # Windows
import cec2014

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
    pathName = "Results/RegPSO"
    makedirs(pathName, exist_ok=True)

    # Initializing result files

    tableFileName = pathName + "/RegPSO_" + str(dims) + "D.csv"

    with open(tableFileName, "w") as resultsFile:

        writer = csv.writer(resultsFile, delimiter = ',')

        writer.writerow( ["Fi_10D", "Best", "Worst", "Median", "Mean", "Std_Dev", "Success_Rate"] )


    for i in funIndexes:

        plotFileName = pathName + "/RegPSO_F" + str(i+1) + "_" + str(dims) + "D_Plot.csv"

        with open(plotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')

            header = ["FES_Multiplier"]
            header.extend( ["Run " + str(i+1) for i in range(numRuns)] )
            header.append("Average")

            writer.writerow( header )

        bestPlotFileName = pathName + "/RegPSO_F" + str(i+1) + "_" + str(dims) + "D_BestPlot.csv"

        with open(bestPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )

        worstPlotFileName = pathName + "/RegPSO_F" + str(i+1) + "_" + str(dims) + "D_WorstPlot.csv"

        with open(worstPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )


        errors = []
        successes = []
        FESResults = []

        for j in range(numRuns):

            # Initialization
            RPSO = RegPSO(functions[i], bounds, popSize=80, clerkK=False, inertiaDecay=True, optimum=optimums[i], prematureThreshold=1.1e-06)
            RPSO.execute()
            results = RPSO.results

            # Treating results

            errors.append(results["errors"][-1])
            successes.append( int(results["errors"][-1] <= RPSO.tol) )
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
                    if(currentFES >= FESThresholds[k] * RPSO.maxFES):
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

            writer.writerow( [str(i+1), min(errors), max(errors), statistics.median(errors),
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
