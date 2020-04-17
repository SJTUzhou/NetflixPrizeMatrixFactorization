import numpy as np
import os
import re
import sys
import time
import ALS_extra_data
from const import *
from matrix_stoc_grad_desc import MatrixModel

class ALS_MatrixModel(MatrixModel):
    def __init__(self):
        MatrixModel.__init__(self)
        # use Alternating-least-square method
        self.userSortedTrainRating = np.load(TEMP_TR_ARRAY_USER)
        self.userSortedCounts = ALS_extra_data.getUserRatingCounts(self.userSortedTrainRating)
        self.lrate = None
        self.feature_num = 100
        self.lmbda = 0.1

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
            if userIdx%1000 == 0:           
                print("Finish the ratings of user {}".format(userIdx))
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
            if movieIdx%100 == 0:           
                print("Finish the ratings of movie {}".format(movieIdx))
        
            


    def training(self, max_epoch, current_epoch=0):
        for epoch in range(current_epoch, max_epoch):
            if epoch != 0:
                self.train_one_epoch()
            self.save_matrix_model(epoch)
            accuracy, rmse = self.evaluat_test()
            self.write_log(epoch, accuracy, rmse)
            # the number of checkpoints that we want to keep
            keep_num = 5
            self.delete_old_model(epoch-keep_num, 1)


    def continue_training(self, max_epoch):
        # Get the current epoch that we have finished training
        with open(self.train_log,"r") as f:
            content = f.read()
            current_epoch_str = re.findall(r'[Ee]poch \d+', content)[-1]
            current_epoch = int(current_epoch_str.split()[-1])
        print("Continue training, loading the checkpoint at Epoch {}".format(current_epoch))
        # Load the latest checkpoints
        self.movieFeature = np.load(MOVIE_FEATURE_FILE.replace("index", str(current_epoch)))
        self.userFeature = np.load(USER_FEATURE_FILE.replace("index", str(current_epoch)))
        # Continue training from the next epoch
        self.training(max_epoch, current_epoch+1)


def main(argv):
    max_epoch = 1200
    ALS_model = ALS_MatrixModel()
    flag = argv[1]
    if flag == 'continue':
        ALS_model.continue_training(max_epoch)
    elif flag == 'start':
        ALS_model.training(max_epoch)
    elif flag == 'delete':
        # Delete the old models manually
        epoch_start = int(argv[2])
        epoch_range = int(argv[3])
        ALS_model.delete_old_model(epoch_start, epoch_range)
    else:
        print("Wrong parameter.")
    
if __name__ == '__main__':
    main(sys.argv)