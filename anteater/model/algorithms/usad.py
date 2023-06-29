#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2023 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/

import copy
import json
import logging
import os
import stat
import time
from itertools import chain
from os import path
from typing import List, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anteater.utils.ts_dataset import TSDataset


class USADConfig:
    """The USADModel model configuration"""

    filename = "usad_config.json"

    def __init__(self, hidden_sizes: Tuple = (25, 10, 5), latent_size: int = 5,
                 dropout_rate: float = 0.25, batch_size: int = 128,
                 num_epochs: int = 250, lr: float = 0.001, step_size: int = 60,
                 window_size: int = 10, weight_decay: float = 0.01, **kwargs):
        """The usad model config initializer"""
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.activation = nn.ReLU
        self.step_size = step_size
        self.window_size = window_size
        self.weight_decay = weight_decay

    @classmethod
    def from_dict(cls, config_dict: dict):
        """class initializer from the config dict"""
        config = cls(**config_dict)
        return config

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """dumps the config dict"""
        config_dict = {}
        for key, val in self.__dict__.items():
            if key == "activation":
                continue
            if hasattr(val, "to_dict"):
                val = val.to_dict()

            config_dict[key] = val

        return config_dict


class USADModel:
    filename = 'usad.pkl'
    config_class = USADConfig

    def __init__(self, config: USADConfig):
        super().__init__()
        self.config = config
        self._num_epochs = config.num_epochs
        self._batch_size = config.batch_size
        self._window_size = config.window_size
        self._lr = config.lr
        self._weight_decay = config.weight_decay
        self.model: USAD = None

    @classmethod
    def load(cls, folder: str, **kwargs):
        """Loads the model from the file"""
        config_file = os.path.join(folder, cls.config_class.filename)
        state_file = os.path.join(folder, cls.filename)

        if not os.path.isfile(config_file) or not os.path.isfile(state_file):
            logging.warning("Unknown model file, load default vae model!")
            config = USADConfig()
            config.update(**kwargs)
            return USADModel(config)

        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(config_file, os.O_RDONLY, modes), "r") as f:
            config_dict = json.load(f)

        config = cls.config_class.from_dict(config_dict)
        model = cls(config=config)

        state_dict = torch.load(state_file)
        if "config" in state_dict:
            state_dict.pop("config")

        for name, val in state_dict.items():
            if hasattr(model, name):
                setattr(model, name, val)

        return model

    def init_model(self, x_dim):
        """Initializes the model"""
        model = USAD(
            x_dim=x_dim,
            config=self.config
        )

        return model

    def train(self, train_values, valid_values):
        self.model = self.model if self.model else self.init_model(train_values.shape[1])

        x_dim = train_values.shape[1] * self._window_size
        train_loader = DataLoader(TSDataset(train_values, self._window_size, self._window_size),
                                  batch_size=self._batch_size,
                                  shuffle=True,
                                  drop_last=True)

        valid_loader = DataLoader(TSDataset(valid_values, self._window_size, self._window_size),
                                  batch_size=self._batch_size)

        train_loss1 = []
        train_loss2 = []

        valid_loss1 = []
        valid_loss2 = []

        # 损失函数
        mse = nn.MSELoss()
        # 优化器
        optimizer_g = torch.optim.Adam(self.model.g_parameters())
        optimizer_d = torch.optim.Adam(self.model.d_parameters())

        train_time = 0
        valid_time = 0
        for epoch in range(self._num_epochs):
            self.model.train()
            train_start = time.time()
            train_losses_g = []
            train_losses_d = []
            for x_batch_train in train_loader:
                # [batch, window, feature] -> [batch, window*feature]
                w = torch.reshape(x_batch_train, (-1, x_dim))
                w_g, w_d, w_g_d = self.model(w)
                beta = np.divide(1, epoch + 1)
                loss_g = beta * mse(w_g, w) + (1 - beta) * mse(w_g_d, w)
                loss_d = beta * mse(w_d, w) - (1 - beta) * mse(w_g_d, w)

                train_losses_g.append(loss_g.detach().numpy())
                train_losses_d.append(loss_d.detach().numpy())

                # 梯度清零
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                # 反向传播
                loss_g.backward(retain_graph=True)
                loss_d.backward()

                # 更新梯度
                optimizer_g.step()
                optimizer_d.step()

            train_loss1_mean = np.mean(train_losses_g)
            train_loss2_mean = np.mean(train_losses_d)
            train_loss1.append(train_loss1_mean)
            train_loss2.append(train_loss2_mean)
            train_time += time.time() - train_start

            self.model.eval()
            val_losses1 = []
            val_losses2 = []
            valid_start = time.time()

            for x_batch_val in valid_loader:
                w = torch.reshape(x_batch_val, (-1, x_dim))
                w_g, w_d, w_g_d = self.model(w)
                beta = np.divide(1, epoch)
                val_loss1 = beta * mse(w, w_g) + (1 - beta) * mse(w, w_g_d)
                val_loss2 = beta * mse(w, w_d) - (1 - beta) * mse(w, w_g_d)

                val_losses1.append(val_loss1.detach().numpy())
                val_losses2.append(val_loss2.detach().numpy())

            valid_time += time.time() - valid_start

            val1_loss = np.mean(val_losses1)
            val2_loss = np.mean(val_losses2)
            valid_loss1.append(val1_loss)
            valid_loss2.append(val2_loss)
            logging.info(f'epoch {epoch} val1_loss: {val1_loss}, val2_loss: {val2_loss}, '
                         f'train_loss1: {train_loss1_mean} train_loss2:{train_loss2_mean}')

        return {
            'train_time': train_time,
            'valid_time': valid_time,
            'train_loss1': np.array(train_loss1).tolist(),
            'train_loss2': np.array(train_loss2).tolist(),
            'valid_loss1': np.array(valid_loss1).tolist(),
            'valid_loss2': np.array(valid_loss2).tolist()
        }

    def predict(self, values):
        x_dim = values.shape[1] * self._window_size
        collect_g, collect_g_d = [], []
        x_loader = DataLoader(TSDataset(values, self._window_size, self._window_size),
                              batch_size=self._batch_size)

        for w in x_loader:
            w = torch.reshape(w, (-1, x_dim))
            w_g, w_d, w_g_d = self.model(w)
            batch_g = torch.reshape(w_g, (-1, self._window_size, values.shape[1]))
            batch_g_d = torch.reshape(w_g_d, (-1, self._window_size, values.shape[1]))

            # 只取最后一个点的重构值
            collect_g.extend(batch_g[:, -1].detach().numpy())
            collect_g_d.extend(batch_g_d[:, -1].detach().numpy())

        return np.array(collect_g), np.array(collect_g_d)

    def save(self, folder):
        """Saves the model into the file"""
        state_dict = {key: copy.deepcopy(val) for key, val in self.__dict__.items()}
        config_dict = self.config.to_dict()

        modes = stat.S_IWUSR | stat.S_IRUSR
        config_file = path.join(folder, self.config_class.filename)
        with os.fdopen(os.open(config_file, os.O_WRONLY | os.O_CREAT, modes), "w") as f:
            f.truncate(0)
            json.dump(config_dict, f, indent=2)

        if "config" in state_dict:
            state_dict.pop("config")

        torch.save(state_dict, os.path.join(folder, self.filename))


class USAD(nn.Module):
    def __init__(self, x_dim, config: USADConfig):
        super().__init__()
        self._shared_encoder = Encoder(x_dim * self._window_size, config)
        self._decoder_g = Decoder(x_dim * self._window_size, config)
        self._decoder_d = Decoder(x_dim * self._window_size, config)

    def g_parameters(self):
        return chain(self._shared_encoder.parameters(), self._decoder_g.parameters())

    def d_parameters(self):
        return chain(self._shared_encoder.parameters(), self._decoder_g.parameters())

    def forward(self, x):
        z = self._shared_encoder(x)
        w_g = self._decoder_g(z)
        w_d = self._decoder_d(z)
        w_g_d = self._decoder_d(self._shared_encoder(w_g))

        return w_g, w_d, w_g_d


class Encoder(nn.Module):
    def __init__(self, x_dim: int, config: USADConfig):
        super().__init__()
        if not config.hidden_sizes:
            hidden_sizes = [x_dim // 2, x_dim // 4]
        else:
            hidden_sizes = config.hidden_sizes

        latent_size = config.latent_size
        dropout_rate = config.dropout_rate
        activation = config.activation

        self.mlp = build_multi_hidden_layers(latent_size, hidden_sizes,
                                             dropout_rate, activation)
        self.linear = nn.Linear(hidden_sizes[-1], latent_size)

    def forward(self, x):
        x = self.mlp(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, x_dim: int, config: USADConfig):
        super().__init__()
        if not config.hidden_sizes:
            hidden_sizes = [x_dim // 4, x_dim // 2]
        else:
            hidden_sizes = config.hidden_sizes[::-1]

        latent_size = config.latent_size
        dropout_rate = config.dropout_rate
        activation = config.activation

        self.mlp = build_multi_hidden_layers(latent_size, hidden_sizes,
                                             dropout_rate, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], x_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.output_layer(x)
        return x


def build_multi_hidden_layers(input_size: int, hidden_sizes: List[int],
                              dropout_rate: float, activation: Callable):
    """build vae model multi-hidden layers"""
    multi_hidden_layers = []
    for i, _ in enumerate(hidden_sizes):
        in_size = input_size if i == 0 else hidden_sizes[i - 1]
        multi_hidden_layers.append(nn.Linear(in_size, hidden_sizes[i]))
        multi_hidden_layers.append(activation())
        multi_hidden_layers.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*multi_hidden_layers)
