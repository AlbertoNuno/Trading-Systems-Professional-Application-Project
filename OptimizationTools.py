import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import tensorflow as tf
from keras.callbacks import  CSVLogger


class CustomCallback(tf.keras.callbacks.Callback):
    # def on_train_batch_end(self, batch, logs=None):
    #     return logs["loss"]
    #
    # def on_test_batch_end(self, batch, logs=None):
    #     return logs["loss"]
    def on_epoch_end(self, epoch, logs=None):
        CSVLogger('log.csv', append=True, separator=';')






def PSO(loss_result,search_space,n_particles,c1,c2):
    #xs = list(y(i) for i in search_space)
        x1p = np.random.random(n_particles)
        x1pL = x1p
        velocidad_x1 = np.zeros(n_particles)
        x1_pg = 0
        fx_pg = 100
        fx_pL = np.ones(n_particles) * fx_pg

    for i in range(0, 1000):
        fx = loss_result
        [val, idx] = fx.min(), fx.idxmin()

        if val.values < float(fx_pg):
            fx_pg = val
            x1_pg = x1p[idx]

        for j in range(0, n_particles):
            if fx.iloc[j].values < fx_pL[j]:
                fx_pL[j] = fx.iloc[j].values
                x1pL = x1p[j]

        velocidad_x1 = velocidad_x1 + c1 * np.random.rand() * (x1_pg - x1p) + c2 * np.random.rand() * (x1pL - x1p)
        x1p = x1p + velocidad_x1

        return [x1_pg,fx_pg]



