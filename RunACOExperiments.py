from models import AntColonyOptimization
from optproblems import cec2005
from os import makedirs
import statistics
import csv

if __name__ == '__main__':

    dims = 10
    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ]
    # functions = [ cec2005.F1(dims), cec2005.F2(dims), cec2005.F3(dims), cec2005.F4(dims), cec2005.F5(dims)]
    functions = [ cec2005.F1(dims) ]
    optimums = [-450, -450, -450, -450, -310]
    FESThresholds = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # numRuns = 25
    numRuns = 1


    # Creating result path
    pathName = "Results/ACO"
    makedirs(pathName, exist_ok=True)

    # Initializing result files

    tableFileName = pathName + "/ACO_" + str(dims) + "D.csv"

    with open(tableFileName, "w") as resultsFile:

        writer = csv.writer(resultsFile, delimiter = ',')

        writer.writerow( ["Fi_10D", "Best", "Worst", "Median", "Mean", "Std_Dev", "Success_Rate"] )


    for i in range(len(functions)):

        plotFileName = pathName + "/ACO_F" + str(i+1) + "_" + str(dims) + "D_Plot.csv"

        with open(plotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')

            header = ["FES_Multiplier"]
            header.extend( ["Run " + str(i+1) for i in range(numRuns)] )
            header.append("Average")

            writer.writerow( header )

        bestPlotFileName = pathName + "/ACO_F" + str(i+1) + "_" + str(dims) + "D_BestPlot.csv"

        with open(bestPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )

        worstPlotFileName = pathName + "/ACO_F" + str(i+1) + "_" + str(dims) + "D_WorstPlot.csv"

        with open(worstPlotFileName, "w") as resultsFile:

            writer = csv.writer(resultsFile, delimiter = ',')
            header = ["FES_Multiplier", "Error"]
            writer.writerow( header )


        errors = []
        successes = []
        FESResults = []

        for j in range(numRuns):

            ACO = AntColonyOptimization(functions[i], bounds, optimum=optimums[i]) # F5: -310 / others: -450
            ACO.execute()
            results = ACO.results

            # Treating results

            errors.append(results["errors"][-1])
            successes.append( int(results["errors"][-1] <= ACO.tol) )
            # True/False values of successes are converted to int, to fetch the mean (success rate)

            # getting errors for different FES values

            currentFES = 0
            currentThresholdIndex = 0
            FESThresholdErrors = [] # errors for each FES multiplier

            for k in range( len( results["FESCounts"] ) ):

                currentFES = results["FESCounts"][k]

                # FES value coincides with the threshold
                if(currentFES == FESThresholds[currentThresholdIndex] * ACO.maxFES):
                    FESThresholdErrors.append(results["errors"][k])
                    print(currentFES, FESThresholds[currentThresholdIndex] * ACO.maxFES, results["errors"][k])
                    print(FESThresholdErrors)

                # last FES count was below the threshold and the current one has passed it
                # add the result for the last FES count
                elif(currentFES > FESThresholds[currentThresholdIndex] * ACO.maxFES):
                    FESThresholdErrors.append(results["errors"][k-1])
                    print(currentFES, FESThresholds[currentThresholdIndex] * ACO.maxFES, results["errors"][k])
                    print(FESThresholdErrors)

                elif ( k == len( results["FESCounts"] ) - 1 ):
                    FESThresholdErrors.append(results["errors"][k])
                    print(currentFES, FESThresholds[currentThresholdIndex] * ACO.maxFES, results["errors"][k])
                    print(FESThresholdErrors)

                currentThresholdIndex += 1


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

            for i in range(len(FESThresholds)):

                row = [FESThresholds[i]]
                rowVals = []
                lastValidVal = 0

                for j in range(numRuns):

                    if( i < len(FESResults[j]) ):
                        row.append(FESResults[j][i])
                        rowVals.append(FESResults[j][i])
                        lastValidVal = FESResults[j][i]

                    else:
                        row.append("None")
                        rowVals.append(lastValidVal) # for the mean, use the last valid error value

                # mn = append(statistics.mean(rowVals))
                # if mn != 0: row.append(mn) # mean different from 0 indicates that there was at least one execution at this threshold
                # else: row.append("None")
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
