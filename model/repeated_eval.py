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


def generate_temp_data():
    start = time.time()
    prepare_tr_val_data()
    sortSaveTrainArrayByUser()
    duration = time.time() - start
    print("Take {:.2f}s to generate temporary data".format(duration))


def running(ALS_index,SGD_index):       
    FEANUM = 50
    LMBDA = 0.03
    MAX_EPOCH = 20
    ALS_model = ALS_MatrixModel(feature_num=FEANUM, lmbda=LMBDA, index=ALS_index)
    p1 = mlp.Process(target=ALS_model.continue_training, args=(MAX_EPOCH,))
    SGD_model = MatrixModel(feature_num=FEANUM, lmbda=LMBDA, lrate=0.05, index=SGD_index)
    p2 = mlp.Process(target=SGD_model.continue_training, args=(MAX_EPOCH,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == "__main__":
    generate_temp_data()
    ALS_index = [i for i in range(100,200)]
    SGD_index = [i for i in range(200,300)]
    for ALS_i, SGD_i in zip(ALS_index,SGD_index):
        running(ALS_i,SGD_i)