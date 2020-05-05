import numpy as np
from copy import deepcopy
import os
import re
import sys
import time
import multiprocessing as mlp
from const import *


class MatrixModel(object):
    def __init__(self, feature_num, lmbda, lrate, index):
        """
        Initialize the matrix model and get all the needed variables and constants for training and validation

        *** Parameters in training  ***
        feature_num(int): the number of factorisation parameters for each film and user
        lrate(float): learning rate
        lmbda(float): the coefficient for regularization

        *** Matrices for rating prediction ***
        movieFeature(2d-array): the movie matrix with shape (feature_num, movie_num)
        userFeature(2d-array): the user matrix with shape (feature_num, user_num)

        *** Datasets formed in numpy ndarray ***
        train_ratings(2d-array): the part of movie_user_ratings(2d-array) used for training
        validation_ratings(2d-array): a part of movie_user_ratings(2d-array) used for validation
        
        *** Auxiliary variables for training and validation ***
        userIds(1d-array): contain the real user IDs, a match between the 0-user_num and the real userIDs
        invUserIdx(dictionary): the inversible match with 'key' the real user Ids and 'item' the user Index ranging from 0 to user_num
        user_offsets(1d-array): user rating preference compared to the global average movie rating
        rating_counts(1d-array): Count the number of rating for each film
        rating_sums(1d-array): Count the sum of all the ratings for each film
        movie_avg_ratings(1d-array): calculated by rating_sums/rating_counts, represents the average rating for each film
        train_rating_counts(1d-array): Count the number of rating for each film in the training dataset
        train_log(str): the file name of the train log
        """
        self.method = "stochastic gradient descent"
        # Parameters in training
        self.feature_num = feature_num
        self.lmbda = lmbda
        self.lrate = lrate
        self.index = index

        # Matrices for rating prediction
        self.userFeature = (2*np.random.rand(self.feature_num, USER_NUM)-1)
        self.movieFeature = (2*np.random.rand(self.feature_num, MOVIE_NUM)-1)

        # Datasets formed in numpy ndarray
        self.train_ratings = np.load(TEMP_TR_ARRAY)
        self.validation_ratings = np.load(TEMP_VAL_ARRAY)

        # Auxiliary variables for training and validation
        self.userIds = np.load(NPY_USER_ID_FILE)
        self.invUserIdx = {self.userIds[i]:i for i in range(USER_NUM)}
        # print("Finish reversing the user index array")
        self.user_offsets = np.load(NPY_USER_OFFSET_FILE)

        self.rating_counts = np.load(NPY_RATING_COUNTS_FILE)
        self.rating_sums = np.load(NPY_RATING_SUMS_FILE)
        self.movie_avg_ratings = self.rating_sums/self.rating_counts
        
        cts = np.load(TEMP_TR_RAT_COUNTS)
        self.train_rating_counts = np.reshape(cts, (cts.shape[0],))
        # print("Train data shape: {}".format(self.train_ratings.shape))
        # print("Validation data shape: {}".format(self.validation_ratings.shape))
        self.train_log = "../logs/train_log_{}.txt".format(self.index)


    def predict_rating(self, movieIdx, userIdx):
        # prediction = movie average rating + user offset + the part of matrix factorization
        return np.sum(self.userFeature[:,userIdx]*self.movieFeature[:,movieIdx]) + self.movie_avg_ratings[movieIdx] + self.user_offsets[userIdx]
        

    def stochastic_grad_desc(self, userIdxes, movieIdx, realRatings):
        for i,userIdx in enumerate(userIdxes):
            error = realRatings[i] - self.predict_rating(movieIdx, userIdx)
            userValue, movieValue = self.userFeature[:,userIdx], self.movieFeature[:,movieIdx]
            userValueCopy = deepcopy(userValue)
            userValue += self.lrate*(error*movieValue - self.lmbda*userValue)
            movieValue += self.lrate*(error*userValueCopy - self.lmbda*movieValue)
            self.userFeature[:,userIdx] = np.clip(userValue,-1.0,1.0)
            self.movieFeature[:,movieIdx] = np.clip(movieValue,-1.0,1.0)
        
    def train_one_epoch(self):
        # use stochastic gradient descent method for training the 2 matrices
        startId = 0
        for movieIdx, rating_num in enumerate(self.train_rating_counts):
            # At each step, extract and train a random batch of the ratings for the same film
            tr_ratings = self.train_ratings[startId:startId+rating_num,:]
            startId += rating_num
            np.random.shuffle(tr_ratings)
            # get the user index from the real user Ids
            realUserIds, realRatings = tr_ratings[:,1], tr_ratings[:,2]
            userIdxes = np.array([self.invUserIdx[Id] for Id in realUserIds])
            self.stochastic_grad_desc(userIdxes, movieIdx, realRatings)
            if movieIdx%5000 == 0:           
                print("Training index {}: Finish the ratings of movie {}".format(self.index, movieIdx))
        

    def training(self, max_epoch, current_epoch=0):
        for epoch in range(current_epoch, max_epoch+1):
            if epoch != 0:
                self.train_one_epoch()
            self.save_matrix_model(self.index, epoch)
            accuracy, rmse = self.evaluat_test()
            self.write_log(epoch, accuracy, rmse)
            # the number of checkpoints that we want to keep
            keep_num = 1
            self.delete_old_model(epoch-keep_num, 1)
            
    
    def continue_training(self, max_epoch):
        if os.path.exists(self.train_log):
            # Get the current epoch that we have finished training
            with open(self.train_log,"r") as f:
                content = f.read()
                current_epoch_str = re.findall(r'epoch \d+', content, flags=re.I)[-1]
                current_epoch = int(current_epoch_str.split()[-1])
                index = re.findall(r'training index ?: ?(\d+)', content, flags=re.I)[0]
            print("Training index {}: Continue training, loading the checkpoint at Epoch {}".format(self.index,current_epoch))
            # Load the latest checkpoints
            self.load_matrix_model(index,current_epoch)
            # Continue training from the next epoch
            self.training(max_epoch, current_epoch+1)
        else:
            self.training(max_epoch, current_epoch=0)

    def load_matrix_model(self, index, epoch):
        self.movieFeature = np.load(MOVIE_FEATURE_FILE.replace("epoch_index", str(index)+'_E'+str(epoch)))
        self.userFeature = np.load(USER_FEATURE_FILE.replace("epoch_index", str(index)+'_E'+str(epoch)))

    def save_matrix_model(self, index, epoch):
        np.save(USER_FEATURE_FILE.replace("epoch_index", str(index)+'_E'+str(epoch)), self.userFeature)
        np.save(MOVIE_FEATURE_FILE.replace("epoch_index", str(index)+'_E'+str(epoch)), self.movieFeature)
    
    def evaluat_test(self):
        # Do the model test and evaluation
        # return accuracy and RMSE(root mean square error)
        error_tol = 0.5
        errors = []
        for movieId, userRealID, real_rating in self.validation_ratings:
            movieIdx = movieId - 1
            userIdx = self.invUserIdx[userRealID]
            predict_rating = self.predict_rating(movieIdx, userIdx)
            predict_rating = np.clip(predict_rating, 1.0, 5.0)
            predict_rating = round(predict_rating)
            error = real_rating - predict_rating
            errors.append(error)
        accuracy = np.mean(np.abs(errors)<error_tol)
        rmse = np.sqrt(np.mean(np.array(np.square(errors))))
        return accuracy, rmse
        
    def write_log(self, epoch, accuracy, rmse):
        with open(self.train_log, 'a+') as f:
            if epoch==0:
                head_str = "dataset index: 1\n\
                    train datasize: {}\n\
                    validation datasize: {}\n\
                    training index: {}\n\
                    method: {}\n\
                    feature_num: {}\n\
                    lrate: {}\n\
                    lmbda: {}\n".format(self.train_ratings.shape[0],self.validation_ratings.shape[0],self.index,\
                        self.method,self.feature_num,self.lrate,self.lmbda)
                f.write(head_str)
            eval_str = "Training index {}: Finish testing Epoch {}, validation accuracy {:.4} with rmse {:.4}\n"\
                .format(self.index, epoch, accuracy, rmse)
            print(eval_str)
            f.write(eval_str)

    def delete_old_model(self, epoch_start, epoch_range):
        epochs = [epoch_start+i for i in range(epoch_range)]
        old_model_files = [USER_FEATURE_FILE.replace("epoch_index", str(self.index)+'_E'+str(i)) for i in epochs]
        old_model_files.extend([MOVIE_FEATURE_FILE.replace("epoch_index", str(self.index)+'_E'+str(i)) for i in epochs])
        for old_file in old_model_files:
            if os.path.exists(old_file):
                os.remove(old_file)
                print("Remove {}".format(old_file))


def run_training(start_tr_index, max_epoch, argv):
    train_dict = {}
    feature_num_list = [5,10,20,50,100,200]
    lmbda_list = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.4]
    last_tr_index = start_tr_index + len(feature_num_list)*len(lmbda_list)
    tr_index = deepcopy(start_tr_index)
    for feature_num in feature_num_list:
        for lmbda in lmbda_list:
            train_dict[tr_index] = {'featureNum':feature_num,'lmbda':lmbda}
            tr_index += 1

    index = int(argv[1])
    if index in [i for i in range(start_tr_index,last_tr_index)]:
        feature_num = train_dict[index]['featureNum']
        lmbda = train_dict[index]['lmbda']
        matrix_model = MatrixModel(feature_num,lmbda,0.05,index)
        flag = argv[2]
        if flag == 'continue':
            matrix_model.continue_training(max_epoch)
        elif flag == 'start':
            matrix_model.training(max_epoch)
        else:
            print("Wrong flag! start or continue ?")
    else:
        print("Wrong training index! Index should be in [{},{})".format(start_tr_index,last_tr_index))
        
    
def single_run(arg_dict):
    index = arg_dict['current_index']
    max_epoch = arg_dict['max_epoch']
    flag = arg_dict['flag']
    feature_num = arg_dict['feature_num']
    lmbda = arg_dict['lmbda']

    matrix_model = MatrixModel(feature_num,lmbda,0.05,index)
    if flag == 'continue':
        matrix_model.continue_training(max_epoch)
    elif flag == 'start':
        matrix_model.training(max_epoch)
    else:
        print("Wrong flag! start or continue ?")


def mlp_run(start_tr_index, max_epoch, flag):
    feature_num_list = [5,10,20,50,100,200]
    lmbda_list = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.4]
    tr_index = deepcopy(start_tr_index)
    arg_dict_list = []
    for feature_num in feature_num_list:
        for lmbda in lmbda_list:
            arg_dict = {
                'current_index':tr_index,
                'feature_num':feature_num,
                'lmbda':lmbda,
                'max_epoch':max_epoch,
                'flag':flag}
            arg_dict_list.append(arg_dict)
            tr_index += 1
    mlp.freeze_support()
    processCores = 8
    with mlp.Pool(processes=processCores) as mPool:
        mPool.map(single_run, arg_dict_list)
        

if __name__ == '__main__':
    SGD_START_INDEX = 1000
    MAX_EPOCH = 20
    # training with single process
    # run_training(SGD_START_INDEX, MAX_EPOCH, sys.argv)
    
    # training with multiple process
    mlp_run(SGD_START_INDEX,MAX_EPOCH,'continue')