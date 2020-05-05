# store the constant variables

# Given files
TRAIN_FILES = ['../netflix-prize-data/combined_data_{}.txt'.format(i) for i in range(1,5)]
MOVIE_TITLE_FILE = '../netflix-prize-data/movie_titles.csv'
PROBE_FILE = '../netflix-prize-data/probe.txt'
QUALIFYING_FILE = '../netflix-prize-data/qualifying.txt'

# constants
MOVIE_NUM = 17770
USER_NUM = 480189
VAL_TR_SPLIT_RATIO = 0.012

# stored temporary checkpoints and traing arrays
USER_FEATURE_FILE = '../ckpt/user_feature_epoch_index.npy'
MOVIE_FEATURE_FILE  = '../ckpt/movie_feature_epoch_index.npy'
TEMP_TR_ARRAY = '../tempData/temp_train_array.npy'
TEMP_VAL_ARRAY = '../tempData/temp_validate_array.npy'
TEMP_TR_RAT_COUNTS = '../tempData/temp_tr_rat_counts.npy'
TEMP_TR_ARRAY_USER = '../tempData/temp_train_array_sorted_by_user.npy' 


CSV_TRAIN_FILE = '../myData/train_data.csv'
NPY_USER_ID_FILE = '../myData/user_ids.npy'
NPY_RATING_FILE = '../myData/observed_ratings.npy'
NPY_RATING_COUNTS_FILE = '../myData/rating_counts.npy'
NPY_RATING_SUMS_FILE = '../myData/rating_sums.npy'
NPY_USER_OFFSET_FILE = '../myData/user_offsets.npy'

"""
README DOC
------------------
observed_ratings.npy
A 2d numpy-array of all the ratings in the ascending order of realMovieID
Shape: (num of ratings in total, 3) with each row represents (realMovieID, realUserID, rating)
Attention! realMovieID and realUserID can be discrete and don’t start from 0.

rating_counts.npy
A 1d numpy-array to store the number of ratings for each film  in the ascending order of movie
Shape: (num of films, )
Attention! Row 0 represents the number of ratings for the Movie 1 and there is not a Movie with its index 0. Row 1 represents the number of ratings for the Movie 2.

rating_sums.npy
A 1d numpy-array to store the sum of all the ratings for one film in the ascending order of movie
Shape: (num of films, )
This file is used to calculate the average rating for each film with the formula 
average_ratings = rating_sums / rating_counts

user_ids.npy
A 1d numpy-array to save all the real user ID in the ascending order.
Shape: (num of users, )
It can be seen as a mapping between the row index (0->num of users) and those real user ID, aiming at extracting one row or one column in the user feature matrix more conveniently.

user_offsets.npy
A 1d array to record the users’ preference compared with the global average rating in the ascending order of userID.
Shape: (num of users, )
For example, if the global user rating is 3.6 and user 1 has a offset 0.2, it means the average rating of user 1 that he gives to those films is 3.6+0.2=3.8. In the matrix factorisation, we can predict with the average rating of a particular movie plus the offset of a particular user as the baseline of the prediction.

temp_train_array.npy
A 2d numpy-array with the same format as observed_ratings.npy
This file is used to train the 2 feature matrices.

temp_validate_array.npy
A 2d numpy-array with the same format as observed_ratings.npy
This file is used to evaluate the 2 feature matrices.

temp_tr_rat_counts.npy
A 1d numpy-array to save the number of ratings of each film in temp_train_array.npy
Shape: (num of films, )
"""

