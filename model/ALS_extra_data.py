import numpy as np
import time
from const import *

def sortSaveTrainArrayByUser(index=None):
    startTime = time.time()
    if index is not None:
        trainArray = np.load(TEMP_TR_ARRAY.replace(".npy","_{}.npy".format(index)))
    else:
        trainArray = np.load(TEMP_TR_ARRAY)
    userSortedTrainArray = trainArray[np.argsort(trainArray[:,1])]
    duration = time.time()-startTime
    if index is not None:
        np.save(TEMP_TR_ARRAY_USER.replace(".npy","_{}.npy".format(index)), userSortedTrainArray)
    else:
        np.save(TEMP_TR_ARRAY_USER, userSortedTrainArray)
    print("Take {:.2f}s to get the train array sorted by user Ids".format(duration))
    return userSortedTrainArray

def getUserRatingCounts(ratingArray):
    startTime = time.time()
    _unique, counts = np.unique(ratingArray[:,1], return_counts=True)
    duration = time.time()-startTime
    print("Take {:.2f}s to get the user rating counts".format(duration))
    return counts

if __name__ == "__main__":
    userSortedTrainArray = sortSaveTrainArrayByUser()
    userRatingCounts = getUserRatingCounts(userSortedTrainArray)
