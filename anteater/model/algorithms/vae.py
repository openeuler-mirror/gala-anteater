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
"""
Time:
Author:
Description: The variational auto-encoder model which will be used to train offline
and online, then predict online.
"""

import copy
import json
import os
import stat
from os import path
from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from anteater.model.algorithms.early_stop import EarlyStopper
from anteater.utils.common import divide
from anteater.utils.log import logger
from anteater.utils.ts_dataset import TSDataset


class VAEConfig:
    """The VAE configuration"""

    filename = "vae_model.json"

    def __init__(
            self,
            hidden_sizes=(25, 10, 5),
            latent_size=5,
            dropout_rate=0.25,
            batch_size=1024,
            num_epochs=30,
            learning_rate=0.001,
            k=1,
            step_size=60,
            num_eval_samples=10,
            **kwargs
    ):
        """The vae config initializer"""
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.activation = nn.ReLU
        self.k = k
        self.step_size = step_size
        self.num_eval_samples = num_eval_samples

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


class VAEModel:
    """The vae-based multivariate time series anomaly
    detector for operating system
    """

    filename = "VAE.pkl"
    config_class = VAEConfig

    def __init__(self, config: VAEConfig, **kwargs):
        """VAEDetector initializer"""
        self.config = copy.copy(config)

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.lr = config.lr

        self.k = config.k
        self.step_size = config.step_size

        self.num_eval_samples = config.num_eval_samples

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None

    @classmethod
    def load(cls, folder: str, **kwargs):
        """Loads the model from the file"""
        config_file = os.path.join(folder, cls.config_class.filename)
        state_file = os.path.join(folder, cls.filename)

        if not os.path.isfile(config_file) or not os.path.isfile(state_file):
            logger.warning("Unknown model file, load default vae model!")
            config = VAEConfig()
            config.update(**kwargs)
            return VAEModel(config)

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
        """Initializing vae model based on data"""
        model = VAE(
            x_dim=x_dim * self.k,
            config=self.config
        )
        return model

    def train(self, x):
        """Start to train model based on training data and validate data"""
        logger.info(f"Using {self.device} device for vae model training")
        self.model = self.model if self.model else self.init_model(x.shape[1])
        self.model.to(self.device)

        x = TSDataset(x, self.k, self.step_size)
        train_size = int(0.7 * len(x))
        val_size = len(x) - train_size
        x_train, x_val = random_split(x, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
        x_loader = DataLoader(x_train, batch_size=self.batch_size, shuffle=True)
        x_val_loader = DataLoader(x_val, batch_size=self.batch_size,
                                  shuffle=True)

        loss_func = nn.MSELoss(reduction='sum')
        early_stopper = EarlyStopper(patience=10)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_batch_count = 0
            for x_batch in x_loader:
                x_batch = x_batch.to(self.device)
                x_batch = torch.transpose(x_batch, 1, 2)
                x_batch = torch.flatten(x_batch, start_dim=1)
                x_batch_hat, means, log_var = self.model(x_batch)
                loss_x_batch_hat = loss_func(x_batch, x_batch_hat)
                loss_kl = -0.5 * torch.mean(
                    torch.sum(1 + log_var - means ** 2 - log_var.exp(), dim=1),
                    dim=0)
                loss = loss_x_batch_hat + loss_kl

                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_batch_count += 1

            self.model.eval()
            val_loss = 0
            val_batch_count = 0
            for x_val_batch in x_val_loader:
                x_val_batch = x_val_batch.to(self.device)
                x_val_batch = torch.transpose(x_val_batch, 1, 2)
                x_val_batch = torch.flatten(x_val_batch, start_dim=1)
                x_val_batch_hat, means, log_var = self.model(x_val_batch)
                loss_recon_x = loss_func(x_val_batch, x_val_batch_hat)
                loss_kl = -0.5 * torch.mean(
                    torch.sum(1 + log_var - means ** 2 - log_var.exp(), dim=1),
                    dim=0)
                loss = loss_recon_x + loss_kl

                val_loss += loss.item()
                val_batch_count += 1

            avg_train_loss = divide(train_loss, train_batch_count)
            avg_valid_loss = divide(val_loss, val_batch_count)

            logger.info(f"Epoch(s): {epoch}\ttrain Loss: {avg_train_loss:.5f}\t"
                        f"validate Loss: {avg_valid_loss:.5f}")

            if early_stopper.early_stop(val_loss):
                logger.info("Early Stopped!")
                break

    def predict(self, x):
        """Detects the x and return reconstruct scores"""
        self.model.eval()
        y = TSDataset(x, self.k, 1)
        y = np.array([y[i] for i in range(len(y))])
        y = torch.FloatTensor(y).to(self.device)
        y = torch.transpose(y, 1, 2)
        y = torch.flatten(y, start_dim=1)

        avg_recon_y = np.zeros(y.shape)
        for _ in range(self.num_eval_samples):
            z_hat, _, _ = self.model(y)
            avg_recon_y += z_hat.cpu().data.numpy()
        avg_recon_y /= self.num_eval_samples

        test_scores = np.abs(avg_recon_y - y.cpu().data.numpy())
        test_scores = test_scores.reshape(-1, x.shape[1], self.k)
        test_scores = np.sum(test_scores, axis=1)

        scores = np.zeros((x.shape[0],), dtype=float)

        scores[:-self.k] = test_scores[:-1, 0].flatten()
        scores[-self.k:] = test_scores[-1]

        return scores

    def fit_transform(self, x):
        """train the variational auto-encoder model based on
        the latest raw data
        """
        logger.info("Start to execute vae model training...")
        x = x.astype(np.float32)

        self.train(x)
        self.model.eval()

        return self.predict(x)

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


class VAE(nn.Module):
    """The variational auto-encoder model implemented by torch"""

    def __init__(self, x_dim, config: VAEConfig):
        """The variational auto-encoder model initializer"""
        super().__init__()

        self.encoder = Encoder(x_dim, config=config)
        self.decoder = Decoder(x_dim, config=config)

    @staticmethod
    def re_parameterize(means, log_var):
        """Re-parameterize the means and vars"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def forward(self, x):
        """The whole pipeline of variational auto-encoder model"""
        means, log_var = self.encoder(x)
        z = self.re_parameterize(means, log_var)
        x_hat = self.decoder(z)

        return x_hat, means, log_var


class Encoder(nn.Module):
    """The vae encoder module"""

    def __init__(self, x_dim: int, config: VAEConfig):
        """The vae encoder module initializer"""
        super().__init__()

        hidden_sizes = config.hidden_sizes
        latent_size = config.latent_size
        dropout_rate = config.dropout_rate
        activation = config.activation

        self.mlp = build_multi_hidden_layers(x_dim, hidden_sizes,
                                             dropout_rate, activation)
        self.linear_means = nn.Linear(hidden_sizes[-1], latent_size)
        self.linear_vars = nn.Linear(hidden_sizes[-1], latent_size)
        self.soft_plus = nn.Softplus()

        nn.init.uniform_(self.linear_vars.weight, -0.01, 0.01)
        nn.init.constant_(self.linear_vars.bias, 0)

    def forward(self, x):
        """The vae encoder module pipeline"""
        x = self.mlp(x)
        means = self.linear_means(x)
        log_vars = self.soft_plus(self.linear_vars(x))
        return means, log_vars


class Decoder(nn.Module):
    """The vae decoder module"""

    def __init__(self, x_dim, config: VAEConfig):
        super().__init__()

        hidden_sizes = config.hidden_sizes[::-1]
        latent_size = config.latent_size
        dropout_rate = config.dropout_rate
        activation = config.activation

        self.mlp = build_multi_hidden_layers(latent_size, hidden_sizes,
                                             dropout_rate, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], x_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """The vae decoder module pipeline"""
        x_hat = self.sigmoid(self.output_layer(self.mlp(z)))
        # remove x_hat = self.output_layer(self.mlp(z))
        return x_hat


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
