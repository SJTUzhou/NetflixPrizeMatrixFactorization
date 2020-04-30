import numpy as np
import os
import re
import sys
import time
import ALS_extra_data
from const import *
from matrix_stoc_grad_desc import MatrixModel, mlp_run

class ALS_MatrixModel(MatrixModel):
    def __init__(self,feature_num,lmbda,index):
        MatrixModel.__init__(self,feature_num,lmbda,"None",index)
        # use Alternating-least-square method
        self.method = "alternating least square"
        self.userSortedTrainRating = np.load(TEMP_TR_ARRAY_USER)
        self.userSortedCounts = ALS_extra_data.getUserRatingCounts(self.userSortedTrainRating)

    # use Alternating-Least-Square to train 2 matrices
    def train_one_epoch(self):
        # train user feature matrix
        startId = 0
        for countNum in self.userSortedCounts:
            tr_ratings = self.userSortedTrainRating[startId:startId+countNum,:]
            userId = tr_ratings[0,1]
            userIdx = self.invUserIdx[userId]
            startId += countNum
            realMovieIds, realRatings = tr_ratings[:,0], tr_ratings[:,2]
            movieIdxes = realMovieIds - 1
            movieMatrix = self.movieFeature[:,movieIdxes]
            # solve least-square problem: a*x = b where a:matrix, x:column vector, b:column vector
            a_damp = np.sqrt(countNum*self.lmbda)*np.eye(self.feature_num)
            a = np.concatenate([movieMatrix.T, a_damp], axis=0)
            b_rating = realRatings-self.user_offsets[userIdx]-self.movie_avg_ratings[movieIdxes]
            b = np.concatenate([b_rating, np.zeros((self.feature_num,))],axis=0)
            x, _res, _rank, _s = np.linalg.lstsq(a,b,rcond=None)
            self.userFeature[:,userIdx] = np.clip(x,-1.0,1.0)
            if userIdx%100000 == 0:           
                print("Training index {}: Finish the ratings of user {}".format(self.index,userIdx))
        # train movie feature matrix
        startId = 0
        for countNum in self.train_rating_counts:
            tr_ratings = self.train_ratings[startId:startId+countNum,:]
            movieId = tr_ratings[0,0]
            movieIdx = movieId - 1
            startId += countNum
            # get the user index from the real user Ids, userIdxes and realRatings are column vector
            realUserIds, realRatings = tr_ratings[:,1], tr_ratings[:,2]
            userIdxes = np.array([self.invUserIdx[Id] for Id in realUserIds])
            # extract the features
            userMatrix = self.userFeature[:,userIdxes]
            # solve least-square problem: a*x = b where a:matrix, x:column vector, b:column vector
            a_damp = np.sqrt(countNum*self.lmbda)*np.eye(self.feature_num)
            a = np.concatenate([userMatrix.T, a_damp], axis=0)
            b_rating = realRatings-self.movie_avg_ratings[movieIdx]-self.user_offsets[userIdxes]
            b = np.concatenate([b_rating,np.zeros((self.feature_num,))], axis=0)
            x, _res, _rank, _s = np.linalg.lstsq(a,b,rcond=None)
            self.movieFeature[:,movieIdx] = np.clip(x,-1.0,1.0)
            if movieIdx%5000 == 0:           
                print("Training index {}: Finish the ratings of movie {}".format(self.index,movieIdx))

def single_run(arg_dict):
    index = arg_dict['current_index']
    max_epoch = arg_dict['max_epoch']
    flag = arg_dict['flag']
    feature_num = arg_dict['feature_num']
    lmbda = arg_dict['lmbda']

    model = ALS_MatrixModel(feature_num,lmbda,index)
    if flag == 'continue':
        model.continue_training(max_epoch)
    elif flag == 'start':
        model.training(max_epoch)
    else:
        print("Wrong flag! start or continue ?")
    
if __name__ == '__main__':
    ALS_START_INDEX = 2000
    MAX_EPOCH = 30

    # training with multiple process
    mlp_run(ALS_START_INDEX,MAX_EPOCH,'start')