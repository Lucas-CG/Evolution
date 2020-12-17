import matplotlib.pyplot as plt
import csv
from os import makedirs
import numpy as np
import locale

colors = ["black", "royalblue", "brown", "gold", "lawngreen", "purple", "orangered", "deeppink"]
# algorithms = ["ABC", "ACO", "AGA", "DE", "GA", "PSO", "RegPSO", "SSO"]
algorithms = ["ABC", "ACO", "AGA", "DE", "GA", "PSO", "SSO"]

meanFES = {}
meanErrors = {}
bestFES = {}
bestErrors = {}
worstFES = {}
worstErrors = {}

for algIndex in range(len(algorithms)):

    meanFES[algorithms[algIndex]] = {}
    meanErrors[algorithms[algIndex]] = {}
    bestFES[algorithms[algIndex]] = {}
    bestErrors[algorithms[algIndex]] = {}
    worstFES[algorithms[algIndex]] = {}
    worstErrors[algorithms[algIndex]] = {}

    for funIndex in range(1, 6):

        pathName = "Results/" + algorithms[algIndex]
        plotFileName = pathName + "/" + algorithms[algIndex] + "_F" + str(funIndex) + "_10D_Plot.csv"
        bestPlotFileName = pathName + "/" + algorithms[algIndex] + "_F" + str(funIndex) + "_10D_BestPlot.csv"
        worstPlotFileName = pathName + "/" + algorithms[algIndex] + "_F" + str(funIndex) + "_10D_WorstPlot.csv"

        meanFES[algorithms[algIndex]][str(funIndex)] = []
        meanErrors[algorithms[algIndex]][str(funIndex)] = []

        with open(plotFileName, "r") as resultsFile:

            reader = csv.DictReader(resultsFile, delimiter = ',')

            for row in reader:
                meanFES[algorithms[algIndex]][str(funIndex)].append(float(row["FES_Multiplier"]))
                meanErrors[algorithms[algIndex]][str(funIndex)].append(float(row["Average"]))

        bestFES[algorithms[algIndex]][str(funIndex)] = []
        bestErrors[algorithms[algIndex]][str(funIndex)] = []

        with open(bestPlotFileName, "r") as resultsFile:

            reader = csv.DictReader(resultsFile, delimiter = ',')

            for row in reader:
                bestFES[algorithms[algIndex]][str(funIndex)].append(float(row["FES_Multiplier"]))
                bestErrors[algorithms[algIndex]][str(funIndex)].append(float(row["Error"]))


        worstFES[algorithms[algIndex]][str(funIndex)] = []
        worstErrors[algorithms[algIndex]][str(funIndex)] = []

        with open(worstPlotFileName, "r") as resultsFile:

            reader = csv.DictReader(resultsFile, delimiter = ',')

            for row in reader:
                worstFES[algorithms[algIndex]][str(funIndex)].append(float(row["FES_Multiplier"]))
                worstErrors[algorithms[algIndex]][str(funIndex)].append(float(row["Error"]))


locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8") # Setting German locale for commas in decimals
plt.rcdefaults()
# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True

# Creating result path
pathName = "Results/Plots"
makedirs(pathName, exist_ok=True)

# Plotting means

for funIndex in range(1, 6):

    for algIndex in range(len(algorithms)):

        plt.plot(meanFES[algorithms[algIndex]][str(funIndex)],
        meanErrors[algorithms[algIndex]][str(funIndex)],
        colors[algIndex], label=algorithms[algIndex])

    plt.title("F" + str(funIndex) + ": Desempenhos Médios")
    plt.xlabel("N * MaxFES")
    plt.ylabel("Erro")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(pathName + "/F" + str(funIndex) + "_Mean.svg", bbox_inches='tight')
    plt.clf()

    for algIndex in range(len(algorithms)):

        plt.plot(bestFES[algorithms[algIndex]][str(funIndex)],
        bestErrors[algorithms[algIndex]][str(funIndex)],
        colors[algIndex], label=algorithms[algIndex])

    plt.title("F" + str(funIndex) + ": Desempenhos das Melhores Rodadas")
    plt.xlabel("N * MaxFES")
    plt.ylabel("Erro")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(pathName + "/F" + str(funIndex) + "_Best.svg", bbox_inches='tight')
    plt.clf()

    for algIndex in range(len(algorithms)):

        plt.plot(worstFES[algorithms[algIndex]][str(funIndex)],
        worstErrors[algorithms[algIndex]][str(funIndex)],
        colors[algIndex], label=algorithms[algIndex])

    plt.title("F" + str(funIndex) + ": Desempenhos das Piores Rodadas")
    plt.xlabel("N * MaxFES")
    plt.ylabel("Erro")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(pathName + "/F" + str(funIndex) + "_Worst.svg", bbox_inches='tight')
    plt.clf()
