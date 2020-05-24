import numpy as np
from tensorflow import keras
from const import *
import os


class KerasMLP:
    def __init__(self, index, reload=False):
        self.inputNum = 50 # 100
        self.batchSize = 50000
        __pre_model_index = 100
        __pre_epoch_index = 20
        self.movieFeature = np.load(MOVIE_FEATURE_FILE.replace("epoch_index", str(__pre_model_index) +
                                                               '_E' + str(__pre_epoch_index)))
        self.userFeature = np.load(USER_FEATURE_FILE.replace("epoch_index", str(__pre_model_index) +
                                                             '_E' + str(__pre_epoch_index)))
        self.index = index
        self.reload = reload
        self.savePath = "../ckpt/keras_mlp_{}.h5".format(index)
        if os.path.exists(self.savePath) and not self.reload:
            os.remove(self.savePath)
        self.logfile = "../keras_logs/mlp_log_{}.txt".format(index)
        if os.path.exists(self.logfile) and not self.reload:
            os.remove(self.logfile)
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

        self.model = self.__get_model()
        if os.path.exists(self.savePath) and self.reload:
            print("Reload saved model")
            self.model.load_weights(self.savePath)

        self.model.summary()

    def __get_model(self):
        model_input = keras.layers.Input((self.inputNum,))
        input_sum = keras.backend.reshape(keras.backend.sum(model_input, axis=1), (-1, 1))
        linear_1 = keras.layers.Dense(128, activation='tanh')(model_input)
        linear_2 = keras.layers.Dense(256, activation='tanh')(linear_1)
        linear_3 = keras.layers.Dense(128, activation='tanh')(linear_2)
        linear_4 = keras.layers.Dense(64, activation='tanh')(linear_3)
        linear_5 = keras.layers.Dense(8, activation='tanh')(linear_4)
        linear_6 = keras.layers.Dense(1)(linear_5)
        output = keras.layers.Add()([input_sum, linear_6])
        model = keras.models.Model(inputs=model_input, outputs=output)
        return model

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
            batch_y = batch_init_data[:, 2] - np.array([self.movieAvgRatings[m]+self.userOffsets[u]
                                                        for m, u in zip(movie_indexs, user_indexs)])
            yield batch_x, batch_y

    def print_to_log(self, content):
        with open(self.logfile, "a+") as f:
            print(content, file=f)

    def train(self):
        optimizer = keras.optimizers.Adadelta(learning_rate=0.1)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
        self.model.fit_generator(self.my_generator(self.tempTrainArray),
                                 steps_per_epoch=self.tempTrainArray.shape[0]//self.batchSize, epochs=1,
                                 validation_data=self.my_generator(self.tempValidationArray),
                                 validation_steps=self.tempValidationArray.shape[0]//self.batchSize,
                                 callbacks=[early_stopping], verbose=1, shuffle=True)
        self.model.save_weights(self.savePath)

    def get_test_data(self, temp_test_data):
        movie_indexs = temp_test_data[:, 0] - 1
        user_indexs = np.array([self.invUserIdx[realId] for realId in temp_test_data[:, 1]])
        test_x = np.array([self.movieFeature[:, m] * self.userFeature[:, u] for m, u in zip(movie_indexs, user_indexs)])
        rating_y = temp_test_data[:, 2]
        base_y = np.array([self.movieAvgRatings[m] + self.userOffsets[u] for m, u in zip(movie_indexs, user_indexs)])
        return test_x, rating_y[:, np.newaxis], base_y[:, np.newaxis]

    def evaluate(self):
        if not self.reload:
            self.model.summary(print_fn=self.print_to_log)
        print("Start model evaluation")
        test_x, rating_y, base_y = self.get_test_data(self.tempValidationArray)

        output_y = self.model.predict(test_x)
        predict_y = output_y + base_y
        predict_y = np.clip(predict_y, 1.0, 5.0)
        predict_y = np.round(predict_y)

        tol = 1e-4
        accuracy = np.mean(np.abs(predict_y - rating_y) < tol)
        test_RMSE = np.sqrt(np.mean(np.array(np.square(predict_y - rating_y))))
        eval_str = "Finish training keras model, test accuracy {:.4f}, test RMSE loss {:.4f}".format(accuracy, test_RMSE)
        print(eval_str)
        self.print_to_log(eval_str)


if __name__ == "__main__":
    # index = {8}
    mlp = KerasMLP(index=8, reload=True)
    mlp.train()
    mlp.evaluate()

