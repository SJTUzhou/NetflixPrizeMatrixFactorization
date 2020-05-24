import numpy as np
from tensorflow import keras
from const import *
import os
import time


class KerasMLP:
    def __init__(self, index, reload=False):
        self.inputNum = 50 # 100
        self.batchSize = 100000
        __pre_model_index = 100
        __pre_epoch_index = 20
        self.movieFeature = np.load(MOVIE_FEATURE_FILE.replace("epoch_index", str(__pre_model_index) +
                                                               '_E' + str(__pre_epoch_index)))
        self.userFeature = np.load(USER_FEATURE_FILE.replace("epoch_index", str(__pre_model_index) +
                                                             '_E' + str(__pre_epoch_index)))
        self.index = index
        self.savePath = "../ckpt/keras_mlp_{}.h5".format(index)
        self.logfile = "../keras_logs/mlp_log_{}.txt".format(index)

        __userIds = np.load(NPY_USER_ID_FILE)
        self.invUserIdx = {__userIds[i]: i for i in range(USER_NUM)}
        self.userOffsets = np.load(NPY_USER_OFFSET_FILE)

        __movieRatingSums = np.load(NPY_RATING_SUMS_FILE)
        __movieRatingCounts = np.load(NPY_RATING_COUNTS_FILE)
        self.movieAvgRatings = __movieRatingSums/__movieRatingCounts

        __pre_dataset_index = 0
        self.tempTrainArray = np.load(TEMP_TR_ARRAY.replace(".npy", "_{}.npy".format(__pre_dataset_index)))
        self.tempValidationArray = np.load(TEMP_VAL_ARRAY.replace(".npy", "_{}.npy".format(__pre_dataset_index)))
        print("train array shape: {}".format(self.tempTrainArray.shape))
        print("test array shape: {}".format(self.tempValidationArray.shape))

        if os.path.exists(self.savePath) and reload:
            print("Reload saved model")
            self.model = keras.models.load_model(self.savePath)
        else:
            self.model = keras.Sequential(
                [
                    keras.layers.Dense(units=64, activation="tanh", input_shape=(self.inputNum, )),
                    keras.layers.Dropout(rate=0.2),
                    # keras.layers.Dense(uits=32, activation="tanh"shape=(self.inputNum,)),
                    # keras.layers.Dropout(rate=0.1),
                    keras.layers.Dense(units=1)
                ]
            )
        self.model.summary()

    def my_generator(self, data_array):
        # print("Start shuffling data")
        # __startTime = time.time()
        # np.random.shuffle(data_array)
        # __duration = time.time() - __startTime
        # print("Take {:.2f}s to shuffle the data".format(__duration))
        counter = 0
        while True:
            batch_init_data = data_array[counter*self.batchSize:(counter+1)*self.batchSize, :]
            movie_indexs = batch_init_data[:, 0] - 1
            user_indexs = np.array([self.invUserIdx[realId] for realId in batch_init_data[:, 1]])
            batch_x = np.array([self.movieFeature[:, m]*self.userFeature[:, u] for m, u in zip(movie_indexs, user_indexs)])
            # batch_x = np.array([np.hstack((self.movieFeature[:, m], self.userFeature[:, u])) for m, u in zip(movie_indexs, user_indexs)])
            batch_y = batch_init_data[:, 2] - np.array([self.movieAvgRatings[m]+self.userOffsets[u]
                                                        for m, u in zip(movie_indexs, user_indexs)])
            yield batch_x, batch_y

    def print_to_log(self, content):
        with open(self.logfile, "a+") as f:
            print(content, file=f)

    def train(self):
        optimizer = keras.optimizers.Adadelta()
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
        self.model.fit_generator(self.my_generator(self.tempTrainArray),
                                 steps_per_epoch=self.tempTrainArray.shape[0]//self.batchSize, epochs=50,
                                 validation_data=self.my_generator(self.tempValidationArray),
                                 validation_steps=self.tempValidationArray.shape[0]//self.batchSize,
                                 callbacks=[early_stopping], verbose=2, shuffle=True)
        self.model.save(self.savePath)

    def get_test_data(self, temp_test_data):
        movie_indexs = temp_test_data[:, 0] - 1
        user_indexs = np.array([self.invUserIdx[realId] for realId in temp_test_data[:, 1]])
        test_x = np.array([self.movieFeature[:, m] * self.userFeature[:, u] for m, u in zip(movie_indexs, user_indexs)])
        # test_x = np.array([np.hstack((self.movieFeature[:, m], self.userFeature[:, u])) for m, u in zip(movie_indexs, user_indexs)])
        test_y = temp_test_data[:, 2]
        base_y = np.array([self.movieAvgRatings[m] + self.userOffsets[u] for m, u in zip(movie_indexs, user_indexs)])
        return test_x, test_y[:, np.newaxis], base_y[:, np.newaxis]

    def evaluate(self):
        self.model.summary(print_fn=self.print_to_log)
        print("Start model evaluation")
        test_x, test_y, base_y = self.get_test_data(self.tempValidationArray)

        output_y = self.model.predict(test_x)
        predict_y = output_y + base_y
        predict_y = np.clip(predict_y, 1.0, 5.0)
        predict_y = np.round(predict_y)

        tol = 1e-4
        accuracy = np.mean(np.abs(predict_y - test_y) < tol)
        test_RMSE = np.sqrt(np.mean(np.array(np.square(predict_y - test_y))))
        eval_str = "Finish training keras model, test accuracy {:.4f}, test RMSE loss {:.4f}".format(accuracy, test_RMSE)
        print(eval_str)
        self.print_to_log(eval_str)


if __name__ == "__main__":
    # index = {0,1,2}
    mlp = KerasMLP(index=2, reload=False)
    mlp.train()
    mlp.evaluate()

