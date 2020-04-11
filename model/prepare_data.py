import numpy as np
import multiprocessing
import csv
from const import *


def read_part_data(file_path):
    ''' Get the data from one text file, Netflix has 4 text files for training.
    parameter:
        file_path: string, the path to the text file
    Return:
        movie_user_ratings(2d-array): an array [[movieId, userId, rating],...]
        userIds: an array of users' Ids
        ratingCounts: an array of the number of ratings for each movie
        ratingSums: an array of the sum of the ratings for each movie
    '''
    with open(file_path, 'r') as f:
        movie_user_ratings = []
        rating_counts = []
        rating_sums = []
        r_count = 0
        r_sum = 0
        lines = f.read().split('\n')
        lines.remove('')
        movieId = 0
        for line in lines:
            if ':' in line:
                if r_count != 0:
                    rating_counts.append(r_count)
                    rating_sums.append(r_sum)
                r_count = 0
                r_sum = 0
                movieId = int(line.replace(':',''))
            else:
                info = line.split(',')
                userId, userRating = int(info[0]), int(info[1])
                movie_user_ratings.append((movieId, userId, userRating))
                r_count += 1
                r_sum += userRating
        else:
            rating_counts.append(r_count)
            rating_sums.append(r_sum)
        movie_user_ratings = np.array(movie_user_ratings)
        userIds = np.unique(movie_user_ratings[:,1])
        rating_counts = np.array(rating_counts)
        rating_sums = np.array(rating_sums)
        return movie_user_ratings, userIds, rating_counts, rating_sums


def read_all_data(train_file_list):
    ''' read train data from a train file list
    Parameter:
        train_file_list: a list containing the path of train files
    Return:
        movie_user_ratings(2d-array): all the data shaped like (movieID, userID, rating) including train part and validation part, extracted from the original Netflix files
        userIds: a sorted array of users' Ids
        ratingCounts: an array of the number of ratings for each movie
        ratingSums: an array of the sum of the ratings for each movie
    '''
    first_file = True
    for file in train_file_list:
        if first_file:
            movie_user_ratings, userIds, rating_counts, rating_sums = read_part_data(file)
            first_file = False
        else:
            ratings, Ids, counts, sums = read_part_data(file)
            movie_user_ratings = np.concatenate([movie_user_ratings, ratings], axis=0)
            userIds = np.concatenate([userIds, Ids], axis=0)
            rating_counts = np.concatenate([rating_counts, counts], axis=0)
            rating_sums = np.concatenate([rating_sums, sums], axis=0)
    userIds = np.unique(userIds)
    np.sort(userIds)
    print("userIds shape: {}".format(userIds.shape))
    print("movie-user-rating matrix shape: {}".format(movie_user_ratings.shape))
    print("movie rating counts shape: {}".format(rating_counts.shape))
    print("movie rating sums shape: {}".format(rating_sums.shape))
    return movie_user_ratings, userIds, rating_counts, rating_sums


def save_user_offsets(movie_user_ratings, userIds):
    ''' Get the user offset (a deviation between each user's average rating and the global average rating)
    parameters:
        movie_user_ratings(2d-array): all the data shaped like (movieID, userID, rating) including train part and validation part, extracted from the original Netflix files
        userIds: a sorted array of users' Ids
    return:
        user_offsets(1d-array): user rating preference compared to the global average movie rating
    '''
    global_rating = np.mean(movie_user_ratings[:,2])
    user_offsets = np.zeros((USER_NUM,), dtype=np.float32)
    user_ratings = {}
    for userId, rating in movie_user_ratings[:,1:]:
        if userId not in user_ratings.keys():
            user_ratings.setdefault(userId, []).append(rating)
        else:
            user_ratings[userId].append(rating)
    for userId, ratings in user_ratings.items():
        user_offsets[np.argwhere(userIds==userId)] = np.mean(np.array(ratings)) - global_rating
        np.save(NPY_USER_OFFSET_FILE, user_offsets)
    return user_offsets 


'''
def write_train_data_into_csv(movie_user_ratings, csv_file):
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        header = ['movieId', 'userId', 'rating']
        writer.writerow(header)
        for i in range(0, movie_user_ratings.shape[0]):
            writer.writerow(movie_user_ratings[i])
'''


def prepare_all_data():
    '''Read and reform the Neflix data and save it in form of numpy array'''
    movie_user_ratings, userIds, rating_counts, rating_sums = read_all_data(TRAIN_FILES)
    print('Finish reading the train data into the numpy arrays')
    save_user_offsets(movie_user_ratings, userIds)
    # write_train_data_into_csv(movie_user_ratings, CSV_TRAIN_FILE)
    # print('Finish writing the train dict into csv file')
    np.save(NPY_RATING_FILE, movie_user_ratings)
    np.save(NPY_USER_ID_FILE, userIds)
    np.save(NPY_RATING_COUNTS_FILE, rating_counts)
    np.save(NPY_RATING_SUMS_FILE, rating_sums)
    print('Finish writing train data into numpy-array files')


def divide_val_tr_data(m_u_ratings, userIds, rat_counts, split_ratio):
    ''' Divide train data with validation data
    Parameters:
        m_u_ratings(2d-array): all the data shaped like (movieID, userID, rating) including train part and validation part, extracted from the original Netflix files
        userIds(1d-array): a sorted array of users' Ids
        rat_counts(1d-array): Count the number of rating for each film
        split_ratio(float): the ratio of the volume of validation data to that of train data
    '''
    train_ratings = None
    validate_ratings = None
    train_rating_counts = None
    # create the training and validation matrices in the order of movie index
    for movieIdx, rating_num in enumerate(rat_counts):
        start = 0 if movieIdx==0 else np.sum(rat_counts[:movieIdx])
        ratings = m_u_ratings[start:start+rating_num, :]
        np.random.shuffle(ratings)
        val_tr_split_idx = int(split_ratio*ratings.shape[0])
        val_ratings = ratings[0:val_tr_split_idx,:]
        tr_ratings = ratings[val_tr_split_idx:,:]
        if train_rating_counts is not None:
            train_rating_counts = np.vstack((train_rating_counts, tr_ratings.shape[0]))
            validate_ratings = np.vstack((validate_ratings, val_ratings))
            train_ratings = np.vstack((train_ratings, tr_ratings))
        else:
            train_rating_counts = np.array(tr_ratings.shape[0])
            validate_ratings = val_ratings
            train_ratings = tr_ratings
    return train_rating_counts, train_ratings, validate_ratings


def prepare_tr_val_data():
    ''' Prepare the train dataset and the validation dataset'''
    movie_user_ratings = np.load(NPY_RATING_FILE)
    userIds = np.load(NPY_USER_ID_FILE)
    rating_counts = np.load(NPY_RATING_COUNTS_FILE)
    tr_rat_counts, tr_ratings, val_ratings = divide_val_tr_data(movie_user_ratings, userIds, rating_counts, VAL_TR_SPLIT_RATIO)
    np.save(TEMP_TR_ARRAY, tr_ratings)
    np.save(TEMP_VAL_ARRAY, val_ratings)
    np.save(TEMP_TR_RAT_COUNTS, tr_rat_counts)
    print("Finish preparing train and validation data")



if __name__ == "__main__":
    prepare_all_data()
    prepare_tr_val_data()





