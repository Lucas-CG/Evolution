from ArtificialBeeColony import ArtificialBeeColony
from optproblems import cec2005

if __name__ == '__main__':

    from optproblems import cec2005
    dims = 10

    bounds = [ [-100 for i in range(dims)], [100 for i in range(dims)] ] # 10-dimensional sphere (optimum: 0)

    # Initialization
    ABC = ArtificialBeeColony(cec2005.F1(dims), bounds, popSize=50, workerOnlookerSplit=0.5, limit=None, numScouts=1, optimum=-450) # F5: -310 / others: -450
    ABC.execute()
    results = ABC.results

    # Treating results

    error = results["errors"][-1]
    success = results["errors"][-1] <= ACO.tol
    generation = results["generations"][-1]
    FESCount = results["FESCounts"][-1]

    # # getting errors for different FES values
    #
    # currentFES = 0
    # currentThresholdIndex = 0
    # FESThresholds = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # FESThresholdErrors = [] # errors for each FES multiplier
    # FESMultipliers = [] # x * MaxFES. Can be a threshold or a custom multiplier (end of execution before MaxFES)
    #
    # for k in range( len( results["FESCounts"] ) ):
    #
    #     currentFES = results["FESCounts"][k]
    #
    #     # FES value coincides with the threshold
    #     if(currentFES == FESThresholds[currentThresholdIndex] * GA.maxFES):
    #         FESThresholdErrors.append(error = results["errors"][k])
    #         FESMultipliers.append(FESThresholds[currentThresholdIndex])
    #         currentThresholdIndex += 1
    #
    #     # last FES count was below the threshold and the current one has passed it
    #     # add the result for the last FES count
    #     elif(currentFES > FESThresholds[currentThresholdIndex] * GA.maxFES):
    #         FESThresholdErrors.append(error = results["errors"][k-1])
    #         FESMultipliers.append(FESThresholds[currentThresholdIndex])
    #         currentThresholdIndex += 1

    print( str(error) + "," + str(success) + "," + str(generation) + "," + str(FESCount) )

    # print("GA: for criterion = " + GA.crit + ", reached optimum of " + str(results["minFits"][-1]) +
    # " (error of " + str(results["errors"][-1]) + ") (points " + str(results["minPoints"][-1]) + ") with " + str(results["generations"][-1]) + " generations" +
    # " and " + str(results["FESCounts"][-1]) + " fitness evaluations" )

    # end = time.time()
    # print("time:" + str(end - start))
