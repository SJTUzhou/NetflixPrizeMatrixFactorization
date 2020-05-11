import numpy as np
from matplotlib import pyplot as plt
import re
import os 
from collections import defaultdict

class TrainRecord():
    def __init__(self,index,method,featureNum,lmbda,rmseList,accuracyList):
        self.index = index
        self.lmbda = lmbda
        self.featureNum = featureNum
        self.method = method
        self.epochRange = [i for i in range(1,len(rmseList))]
        self.accuracy = accuracyList[1:]
        self.rmse = rmseList[1:]
        self.lastRmse = rmseList[-1]


def formatTrainRecords(logFiles):
    trainRecords = []
    for logfile in logFiles:
        with open(logfile,'r') as log:
            content = log.read()
            index = int(re.findall(r'training index ?: ?(\d+)\n',content,flags=re.I)[0])
            method = re.findall(r'method ?: ?(.*)\n',content,flags=re.I)[0]
            method = "SGD" if "grad" in method else "ALS"
            featureNum = int(re.findall(r'feature_num ?: ?(\d+)',content,flags=re.I)[0])
            lmbda = float(re.findall(r'la?mbda ?: ?(\d+\.?\d*)',content,flags=re.I)[0])
            rmse_list = [float(i) for i in re.findall(r'rmse (\d+\.?\d*)',content,flags=re.I)]
            accuracy_list = [float(i) for i in re.findall(r'accuracy ?(\d+\.?\d*)', content,flags=re.I)]
            trainRecords.append(TrainRecord(index,method,featureNum,lmbda,rmse_list,accuracy_list))
    return trainRecords


def plotRecordEvolution(trainRecords):
    ''' Plot the rmse evolution for each given training record
    '''
    xmin, xmax = 0, 32
    def settings(ylabel):
        plt.xlim((xmin,xmax))
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(ylabel)
    trivialRMSE = 0.9691
    rmseTicks = [0.010*i for i in range(87,97)] + [trivialRMSE]
    trivialAccuracy = 0.4285
    plt.subplot(121)
    plt.hlines(trivialRMSE,xmin,xmax,'r',"dashed",label="trivial RMSE")
    for record in trainRecords:
        label = "{}(feature number: {}, lambda: {})".format(record.method,record.featureNum,record.lmbda)
        plt.plot(record.epochRange, record.rmse,'^-',label=label)
    plt.yticks(rmseTicks)
    settings('RMSE')
    plt.subplot(122)
    plt.hlines(trivialAccuracy,xmin,xmax,'r',"dashed",label="trivial accuracy")
    for record in trainRecords:
        label = "{}(feature number: {}, lambda: {})".format(record.method,record.featureNum,record.lmbda)
        plt.plot(record.epochRange, record.accuracy,'^-',label=label)
    settings('Accuracy')
    fig = plt.gcf()
    fig.set_size_inches((16,9),forward=False)
    fig.savefig("../pic/trainLogEvolutions.png",dpi=500)
    plt.show()

def plotRecordResult(trainRecords):
    ''' Plot the reached rmse with parameter lambda and feature number
    '''
    trivialRMSE = 0.9691
    lambdaTicks = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.4]
    rmseTicks = [0.010*i for i in range(87,97)] + [trivialRMSE]
    recordDict = defaultdict(list)
    for record in trainRecords:
        myKey = record.method + '_Feature_Num_' + str(record.featureNum)
        recordDict[myKey].append(record)
    for recordKey, recordList in recordDict.items():
        xLambda = [record.lmbda for record in recordList]
        yRMSE = [record.lastRmse for record in recordList]
        plt.plot(xLambda, yRMSE, '^-', label=recordKey)
    plt.hlines(trivialRMSE,0,0.4,'r',"dashed",label="trivial RMSE")
    plt.xlabel("lambda (Coefficient of regularisation L_2)")
    plt.xticks(lambdaTicks)
    plt.yticks(rmseTicks)
    plt.ylabel("RMSE (Root Mean Square Error)")
    plt.xlim((-0.005,0.405))
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches((20,9),forward=False)
    fig.savefig("../pic/trainLogResults.png",dpi=500)
    plt.show()



if __name__=="__main__":
    logDir = "../logs/"
    logFiles = [os.path.join(logDir,mF) for mF in os.listdir(logDir) if os.path.isfile(os.path.join(logDir,mF))]
    trainRecords = formatTrainRecords(logFiles)
    plotRecordResult(trainRecords)

    selectedIndex = [1000,2000,1016,2016]
    selectedRecords = []
    for record in trainRecords:
        if record.index in selectedIndex:
            selectedRecords.append(record)
    plotRecordEvolution(selectedRecords)
