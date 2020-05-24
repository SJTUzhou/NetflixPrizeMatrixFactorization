import numpy as np
import os
# if continue training, uncomment the following line
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
from const import *


class KerasMLP:
    def __init__(self, index, reload=False):
        self.batchSize = 24000
        self.userVecLen = 20
        self.movieVecLen = 60
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

        self.reload = reload
        self.model = self.__get_model()
        if os.path.exists(self.savePath) and self.reload:
            print("Reload saved model")
            self.model.load_weights(self.savePath)
        self.model.summary()

    def __get_model(self):
        movie_input = keras.layers.Input((1,))
        user_input = keras.layers.Input((1,))
        movie_embed = keras.layers.Embedding(input_dim=MOVIE_NUM, output_dim=self.movieVecLen, input_length=1)(movie_input)
        user_embed = keras.layers.Embedding(input_dim=USER_NUM, output_dim=self.userVecLen, input_length=1)(user_input)
        merged = keras.layers.Concatenate()([movie_embed, user_embed])
        flatten = keras.layers.Flatten()(merged)
        linear_1 = keras.layers.Dense(128, activation='tanh')(flatten)
        linear_2 = keras.layers.Dense(256, activation='tanh')(linear_1)
        linear_3 = keras.layers.Dense(512, activation='tanh')(linear_2)
        linear_4 = keras.layers.Dense(128, activation='tanh')(linear_3)
        linear_5 = keras.layers.Dense(64, activation='tanh')(linear_4)
        output = keras.layers.Dense(1)(linear_5)
        model = keras.models.Model(inputs=[movie_input, user_input], outputs=output)
        return model

    def get_m_u_r_b_data(self, data_array):
        x_movie_indexs = data_array[:, 0] - 1
        x_user_indexs = np.array([self.invUserIdx[realId] for realId in data_array[:, 1]])
        baselines = np.array([self.movieAvgRatings[m]+self.userOffsets[u] for m, u in zip(x_movie_indexs, x_user_indexs)])
        y_ratings = data_array[:, 2]
        # return with array shape: (num, 1)
        return x_movie_indexs[:, np.newaxis], x_user_indexs[:, np.newaxis], y_ratings[:, np.newaxis], baselines[:, np.newaxis]

    def print_to_log(self, content):
        with open(self.logfile, "a+") as f:
            print(content, file=f)

    def train(self):
        optimizer = keras.optimizers.Adadelta(learning_rate=0.1)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        checkpoint_filepath = '../ckpt/keras_mlp_{}.h5'.format(self.index)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='auto',
            save_best_only=True)

        tr_movie, tr_user, tr_rating, tr_base = self.get_m_u_r_b_data(self.tempTrainArray)
        val_movie, val_user, val_rating, val_base = self.get_m_u_r_b_data(self.tempValidationArray)
        self.model.fit(x=[tr_movie, tr_user], y=tr_rating, batch_size=self.batchSize, epochs=100, verbose=2,
                       validation_data=([val_movie, val_user], val_rating), shuffle=True,
                       callbacks=[early_stopping, model_checkpoint_callback])
        self.model.save_weights(self.savePath)

    def evaluate(self):
        if not self.reload:
            self.model.summary(print_fn=self.print_to_log)
        print("Start model evaluation")
        val_movie, val_user, val_rating, val_base = self.get_m_u_r_b_data(self.tempValidationArray)

        output_y = self.model.predict([val_movie, val_user])
        # predict_y = output_y + val_base
        predict_rating = np.round(np.clip(output_y, 1.0, 5.0))

        tol = 1e-4
        accuracy = np.mean(np.abs(predict_rating - val_rating) < tol)
        test_RMSE = np.sqrt(np.mean(np.array(np.square(predict_rating - val_rating))))
        eval_str = "Finish training keras model, test accuracy {:.4f}, test RMSE loss {:.4f}\n".format(accuracy, test_RMSE)
        print(eval_str)
        with open(self.logfile, 'a+') as f:
            f.write(eval_str)


if __name__ == "__main__":
    # index = {3,4,5,6,7}
    mlp = KerasMLP(index=7, reload=False)
    if not mlp.reload:
        mlp.train()
    mlp.evaluate()

