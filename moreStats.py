import statistics as stats
import matplotlib.pyplot as plt
import math
import numpy as np
from os import listdir
import csv
class Model():
    def __init__(self, width, num, input):
        self.width = width
        self.num = num
        self.input = input
        assert width >= input

if __name__ == "__main__":
    inputWidths = {"addnist":64,"flowers":24,"goodvbad":48,"rice":5,"mnist":5}
    models = {
        "addnist":{
        },
        "flowers":{
        },
        "goodvbad":{
        },
        "rice":{
        },
        "mnist":{
        }


    }

    names = ["addnist","flowers","goodvbad","rice","mnist"]
    names = ["addnist"]
    for name in names:
        exprs = {}
        trainMeans = {}
        testMeans = {}
        for file in listdir(f"moreResults/{name}"):
            width, depth = file.replace(".csv", "").split("_")
            width = int(width)
            depth = int(depth)
            if depth > 6:
                continue
            try:
                exprs[file] = ((width / inputWidths[name]) ** ((depth - 1) * inputWidths[name])) * (
                            width ** inputWidths[name])
            except OverflowError:
                print("infinity")
                continue

            trainAcc = []
            testAcc = []
            fileIn = open(f"moreResults/{name}/{file}")
            reader = csv.DictReader(fileIn)
            entries = [line.replace("\n","").split(",") for line in fileIn.readlines()[1:] if line != '\n']
            #print(entries)
            fileIn.close()
            for entry in entries:
                trainAcc.append(round(float(entry[0]),4))
                testAcc.append(round(float(entry[1]),4))
            #print(trainAcc)
            #print(testAcc)
            #print (trainAcc)
            #print(testAcc)
            print(file)
            trainMean = round(stats.mean(trainAcc),4)

            testMean = round(stats.mean(testAcc),4)

            trainMeans[file] = trainMean
            testMeans[file] = testMean
            print(trainMean)
            print(testMean)
            print(exprs[file])
            print()

        trainMeansPlot = np.array([])
        testMeansPlot = np.array([])
        exprsPlot = np.array([])
        for x in trainMeans.values():
            trainMeansPlot = np.append(trainMeansPlot,x)
        for x in testMeans.values():
            testMeansPlot = np.append(testMeansPlot, x)
        for x in exprs.values():
            exprsPlot = np.append(exprsPlot,math.log10(x))
        plt.title(name)
        plt.xlabel("Log expressivity")
        plt.ylabel("Mean accuracy")
        plt.scatter(exprsPlot,trainMeansPlot,c="blue",label="Train")
        plt.scatter(exprsPlot,testMeansPlot,c="orange",label="Test")
        plt.legend(loc="upper left")
        plt.show()
