#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/

import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model, regularizers
from anteater.utils.log import logger


class Encoder(Model):
    def __init__(self, input_dims: int, z_dims: int, nn_size=None):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 2, input_dims // 4)

        self.model = Sequential()
        for cur_size in nn_size:
            self.model.add(layers.Dense(cur_size, activation="relu"))

        self.model.add(layers.Dense(z_dims, activation="relu",
                       activity_regularizer=regularizers.l2(0.004)))
        self.model.add(layers.Dropout(0.5))

    def call(self, x):
        z = self.model(x)
        return z


class Decoder(Model):
    def __init__(self, input_dims: int, z_dims: int, nn_size=None):
        super().__init__()

        if not nn_size:
            nn_size = (input_dims // 4, input_dims // 2)

        self.model = Sequential()
        for cur_size in nn_size:
            self.model.add(layers.Dense(cur_size, activation="relu"))

        self.model.add(layers.Dense(input_dims))

    def call(self, z):
        w = self.model(z)
        return w


class SlidingWindowDataset():

    def __init__(self, values, window_size):
        self._values = values
        self._window_size = window_size
        self.strided_values = self._to_windows(self._values)

    def __getitem__(self, index):
        return np.copy(self.strided_values[index]).astype(np.float32)

    def __len__(self):
        return np.size(self.strided_values, 0)

    def _to_windows(self, values):
        """Divides data by time window"""
        sliding_windows = []
        for i in range(values.shape[0] - self._window_size + 1):
            sliding_windows.append(self._values[i:i + self._window_size])
        return np.array(sliding_windows)


class SlidingWindowDataLoader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

        if self._shuffle:
            self._idxs = np.random.permutation(
                self._dataset.shape[0])
        else:
            self._idxs = np.arange(self._dataset.shape[0])

        if self._drop_last:
            self.total = len(self._dataset) // self._batch_size
        else:
            self.total = (len(self._dataset) +
                          self._batch_size - 1) // self._batch_size

    def get_item(self, idx):
        """Gets batch data"""
        if (idx + 1) * self._batch_size > self._dataset.shape[0]:
            batch_idx = self._idxs[idx * self._batch_size:]
            batch = self._dataset[batch_idx]
        else:
            batch_idx = self._idxs[idx *
                                   self._batch_size:(idx + 1) * self._batch_size]
            batch = self._dataset[batch_idx]

        return batch


class OnlineAd():
    def __init__(self, x_dims: int, max_epochs: int = 250, batch_size: int = 128,
                 encoder_nn_size=None, decoder_nn_size=None,
                 z_dims: int = 38, window_size: int = 10):

        self._x_dims = x_dims
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._encoder_nn_size = encoder_nn_size
        self._decoder_nn_size = decoder_nn_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._step = 0

        self._shared_encoder = Encoder(
            input_dims=self._input_dims, z_dims=self._z_dims)
        self._decoder_g = Decoder(
            z_dims=self._z_dims, input_dims=self._input_dims)
        self._decoder_d = Decoder(
            z_dims=self._z_dims, input_dims=self._input_dims)

    @staticmethod
    def get_train_loss(epoch, w, w_g, w_d, w_g_d):
        mse = tf.keras.losses.MeanSquaredError()
        loss_g = (1 / epoch) * mse(w_g, w) + (1 - 1 / epoch) * mse(w_g_d, w)
        loss_d = (1 / epoch) * mse(w_d, w) - (1 - 1 / epoch) * mse(w_g_d, w)

        return loss_g, loss_d

    @staticmethod
    def get_valid_loss(epoch, w, w_g, w_d, w_g_d):
        mse = tf.keras.losses.MeanSquaredError()
        loss_g = 1 / epoch * mse(w, w_g) + (1 - 1 / epoch) * mse(w, w_g_d)
        loss_d = 1 / epoch * mse(w, w_d) - (1 - 1 / epoch) * mse(w, w_g_d)

        return loss_g, loss_d

    @staticmethod
    def fit_grad(grad_e, grad_g, grad_d, grad_ae_g, grad_ae_d):
        for i, x in enumerate(grad_ae_g):
            if i < len(grad_ae_g) // 3:
                grad_e.append(x + grad_ae_d[i])
            elif i < 2 * len(grad_ae_g) // 3:
                grad_g.append(x + grad_ae_d[i])
            else:
                grad_d.append(x + grad_ae_d[i])
        return grad_e, grad_g, grad_d

    def load_sliding_window(self, values, **kwargs):
        shuffle_ = kwargs.get("shuffle", 'Not exist')
        drop_last_ = kwargs.get("drop_last", 'Not exist')

        if shuffle_ == 'Not exist' and drop_last_ == 'Not exist':
            res = SlidingWindowDataLoader(
                SlidingWindowDataset(values, self._window_size).strided_values,
                batch_size=self._batch_size
            )

        else:
            res = SlidingWindowDataLoader(
                SlidingWindowDataset(values, self._window_size).strided_values,
                batch_size=self._batch_size, shuffle=shuffle_, drop_last=drop_last_
            )

        return res

    def get_coder(self, w):
        z = self._shared_encoder(w)
        w_g = self._decoder_g(z)
        w_d = self._decoder_d(z)
        w_g_d = self._decoder_d(self._shared_encoder(w_g))
        return z, w_g, w_d, w_g_d

    def fit(self, train_values, valid_values, alpha=0.5):
        """Start to train model based on training data and validate data"""
        train_sliding_window = self.load_sliding_window(train_values, shuffle=True, drop_last=True)
        valid_sliding_window = self.load_sliding_window(valid_values)
        train_loss1, train_loss2, valid_loss1, valid_loss2 = [], [], [], []
        optimizer_g = tf.keras.optimizers.Adam(lr=0.001)
        optimizer_d = tf.keras.optimizers.Adam(lr=0.001)
        train_time, valid_time = 0, 0
        for epoch in range(1, self._max_epochs + 1):
            train_start = time.time()
            train_losses1, train_losses2 = [], []
            for step in range(train_sliding_window.total):
                grad_e, grad_g, grad_d = [], [], []
                with tf.GradientTape(persistent=True) as tape:
                    x_batch_train = train_sliding_window.get_item(step)
                    w = tf.reshape(x_batch_train, (-1, self._input_dims))
                    z, w_g, w_d, w_g_d = self.get_coder(w)
                    loss_g, loss_d = self.get_train_loss(epoch, w, w_g, w_d, w_g_d)
                train_losses1.append(loss_g.numpy())
                train_losses2.append(loss_d.numpy())
                encoder_train_var = self._shared_encoder.trainable_variables
                decode_g_train_var = self._decoder_g.trainable_variables
                decode_d_train_var = self._decoder_d.trainable_variables
                grad_ae_g = tape.gradient(loss_g, encoder_train_var + decode_g_train_var + decode_d_train_var)
                grad_ae_d = tape.gradient(loss_d, encoder_train_var + decode_g_train_var + decode_d_train_var)
                grad_e, grad_g, grad_d = self.fit_grad(grad_e, grad_g, grad_d, grad_ae_g, grad_ae_d)
                optimizer_g.apply_gradients(zip(grad_e + grad_g, encoder_train_var + decode_g_train_var))
                optimizer_d.apply_gradients(zip(grad_e + grad_d, encoder_train_var + decode_d_train_var))
                del tape
            train_loss1.append(np.mean(train_losses1))
            train_loss2.append(np.mean(train_losses2))
            train_time += time.time() - train_start
            val_losses1, val_losses2 = [], []
            valid_start = time.time()
            for step in range(valid_sliding_window.total):
                x_batch_val = valid_sliding_window.get_item(step)
                w = tf.reshape(x_batch_val, (-1, self._input_dims))
                z, w_g, w_d, w_g_d = self.get_coder(w)
                val_loss1, val_loss2 = self.get_valid_loss(epoch, w, w_g, w_d, w_g_d)
                val_losses1.append(val_loss1.numpy())
                val_losses2.append(val_loss2.numpy())
            valid_time += time.time() - valid_start
            valid_loss1.append(np.mean(val_losses1))
            valid_loss2.append(np.mean(val_losses2))
            logger.info(f"Epoch: {epoch}\tval_loss1: {np.mean(val_losses1):.5f}\tval_loss2: {np.mean(val_losses2):.5f}"
                        f"\ttrain_loss1: {np.mean(train_losses1):.5f}\t train_loss2: {np.mean(train_losses2):.5f}\t")

        return {
            'train_time': train_time, 'valid_time': valid_time,
            'train_loss1': np.array(train_loss1).tolist(), 'train_loss2': np.array(train_loss2).tolist(),
            'valid_loss1': np.array(valid_loss1).tolist(), 'valid_loss2': np.array(valid_loss2).tolist()
        }

    def predict(self, values):
        """Detects the values"""
        collect_g = []
        collect_g_d = []
        data_sliding_window = self.load_sliding_window(values)
        for step in range(data_sliding_window.total):
            w = data_sliding_window.get_item(step)
            w = tf.reshape(w, (-1, self._input_dims))

            z = self._shared_encoder(w)
            w_g = self._decoder_g(z)
            w_d = self._decoder_d(z)
            w_g_d = self._decoder_d(self._shared_encoder(w_g))

            batch_g = tf.reshape(w_g, (-1, self._window_size, self._x_dims))
            batch_g_d = tf.reshape(
                w_g_d, (-1, self._window_size, self._x_dims))

            collect_g.extend(batch_g[:, -1])
            collect_g_d.extend(batch_g_d[:, -1])

        return np.array(collect_g), np.array(collect_g_d)

    def save(self, shared_encoder_path, decoder_g_path, decoder_d_path):
        """Saves the model"""
        self._shared_encoder.save_weights(shared_encoder_path)
        self._decoder_g.save_weights(decoder_g_path)
        self._decoder_d.save_weights(decoder_d_path)

    def restore(self, shared_encoder_path, decoder_g_path, decoder_d_path):
        """Loads the model"""
        self._shared_encoder.load_weights(shared_encoder_path)
        self._decoder_g.load_weights(decoder_g_path)
        self._decoder_d.load_weights(decoder_d_path)

    def save_weights(self):
        """Saves the weights"""
        encoder = self._shared_encoder.get_weights()
        decoder_d = self._decoder_d.get_weights()
        decoder_g = self._decoder_g.get_weights()
        return (encoder, decoder_d, decoder_g)

    def restore_weights(self, encoder, decoder_d, decoder_g):
        """Loads the weights"""
        self._shared_encoder.set_weights(encoder)
        self._decoder_d.set_weights(decoder_d)
        self._decoder_g.set_weights(decoder_g)
