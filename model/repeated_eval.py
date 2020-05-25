import numpy as np
from copy import deepcopy
import os
import re
import sys
import time
import multiprocessing as mlp
from const import *
from matrix_stoc_grad_desc import MatrixModel
from ALS_matrix import ALS_MatrixModel
from prepare_data import divide_val_tr_data, prepare_tr_val_data
from ALS_extra_data import sortSaveTrainArrayByUser


def generate_temp_data(index):
    start = time.time()
    prepare_tr_val_data(index)
    sortSaveTrainArrayByUser(index)
    duration = time.time() - start
    print("Take {:.2f}s to generate temporary data".format(duration))


def runningALS(indexDict):
    # parallel on CPUs by default
    ALS_index = indexDict['ALS']
    datasetIndex = indexDict['DATASET']        
    FEANUM = 50
    LMBDA = 0.03
    MAX_EPOCH = 20
    ALS_model = ALS_MatrixModel(feature_num=FEANUM, lmbda=LMBDA, index=ALS_index, datasetIndex=datasetIndex)
    ALS_model.continue_training(MAX_EPOCH)
    

def runningSGD(indexDict):
    SGD_index = indexDict['SGD']
    datasetIndex = indexDict['DATASET'] 
    FEANUM = 50
    LMBDA = 0.03
    MAX_EPOCH = 20
    SGD_model = MatrixModel(feature_num=FEANUM, lmbda=LMBDA, lrate=0.05, index=SGD_index, datasetIndex=datasetIndex)
    SGD_model.continue_training(MAX_EPOCH)


def main(startIndex):
    times = 8
    datasetIndexList = [i for i in range(startIndex, startIndex+times)]
    mlp.freeze_support()
    with mlp.Pool(processes=8) as mPool:
        mPool.map(generate_temp_data, datasetIndexList)
    
    ALS_index = [i+100 for i in datasetIndexList]
    SGD_index = [i+200 for i in datasetIndexList]

    for ALS_i,dataset in zip(ALS_index,datasetIndexList):
        runningALS({'ALS':ALS_i,'DATASET':dataset})
    
    argDictList = [{'SGD':SGD_i,'DATASET':dataset} for SGD_i,dataset in zip(SGD_index,datasetIndexList)]
    for argDict in argDictList:
        runningSGD(argDict)

if __name__ == "__main__":
    startIndex = 0 # startIndex = N * CPU_NUM, with N an integer >= 0
    main(startIndex)