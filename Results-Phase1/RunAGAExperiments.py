from models import AdaptiveGA
from optproblems import cec2005
from os import makedirs
import statistics
import csv

if __name__ == '__main__':

    dims = 10
    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ]
    functions = [ cec2005.F1(dims), cec2005.F2(dims), cec2005.F3(dims), cec2005.F4(dims), cec2005.F5(dims)]
    optimums = [-450, -450, -450, -450, -310]
    FESThresholds = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    numRuns = 25


    # Creating result path
    pathName = "Results/AGA"
    makedirs(pathName, exist_ok=True)

    # Initializing result files

    tableFileName = pathName + "/AGA_" + str(dims) + "D.csv"

    with open(tableFileName, "w") as resultsFile:

        writer = csv.writer(resultsFile, delimiter = ',')

        writer.writerow( ["Fi_10D", "Best", "Worst", "Median", "Mean", "Std_Dev", "Success_Rate"] )


    for i in range(len(functions)):

        plotFileName = pathName + "/AGA_F" + str(i+1) + "_" + str(dims) + "D_Plot.csv"

        with open(plotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')

            header = ["FES_Multiplier"]
            header.extend( ["Run " + str(i+1) for i in range(numRuns)] )
            header.append("Average")

            writer.writerow( header )

        bestPlotFileName = pathName + "/AGA_F" + str(i+1) + "_" + str(dims) + "D_BestPlot.csv"

        with open(bestPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )

        worstPlotFileName = pathName + "/AGA_F" + str(i+1) + "_" + str(dims) + "D_WorstPlot.csv"

        with open(worstPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )


        errors = []
        successes = []
        FESResults = []

        for j in range(numRuns):

            AGA = AdaptiveGA(functions[i], bounds, crit="min", optimum=optimums[i], tol=1e-08, eliteSize=0, matingPoolSize=70, popSize=70, adaptiveEpsilon=1e-05) #F5 = -310

            AGA.setParentSelection(AGA.tournamentSelection, (True,) )
            AGA.setCrossover(AGA.blxAlphaCrossover, (0.5, 1)) # alpha, prob
            AGA.setMutation(AGA.adaptiveCreepMutation, (1,)) # prob
            AGA.setNewPopSelection(AGA.genitor, None)
            AGA.execute()
            results = AGA.results


            # Treating results

            errors.append(results["errors"][-1])
            successes.append( int(results["errors"][-1] <= AGA.tol) )
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
                    if(currentFES >= FESThresholds[k] * AGA.maxFES):
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
