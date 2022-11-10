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
from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from anteater.model.algorithms.early_stopper import EarlyStopper
from anteater.model.base import DetectorConfig, DetectorBase
from anteater.utils.log import logger


class VAEConfig(DetectorConfig):
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
            **kwargs
    ):
        """The vae config initializer"""
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.activation = nn.ReLU


class VAEDetector(DetectorBase):
    """The vae-based multivariate time series anomaly
    detector for operating system

    """

    filename = "VAE.pkl"
    config_class = VAEConfig

    def __init__(self, config: VAEConfig, **kwargs):
        """VAEDetector initializer"""
        super().__init__(config)
        self.config = copy.copy(config)
        self.hidden_sizes = config.hidden_sizes
        self.latent_size = config.latent_size
        self.dropout_rate = config.dropout_rate
        self.activation = config.activation

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.lr = config.lr

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None

    @property
    def post_processes(self):
        return self.config.post_processes()

    def init_model(self, x_dim):
        """Initializing vae model based on data"""
        model = VAE(
            x_dim=x_dim,
            config=self.config
        )
        return model

    def train(self, x, x_val):
        """Start to train model based on training data and validate data"""
        logger.info(f"Using {self.device} device for vae model training")
        self.model = self.model if self.model else self.init_model(x.shape[1])

        x_loader = DataLoader(x, batch_size=self.batch_size, shuffle=True)
        x_val_loader = DataLoader(x_val, batch_size=self.batch_size,
                                  shuffle=True)

        loss_func = nn.MSELoss()
        early_stopper = EarlyStopper()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_batch_count = 0
            for x_batch in x_loader:
                x_batch = x_batch.to(self.device)
                x_batch_hat, means, log_var, _ = self.model(x_batch)
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
                x_val_batch_hat, means, log_var, _ = self.model(x_val_batch)
                loss_recon_x = loss_func(x_val_batch, x_val_batch_hat)
                loss_kl = -0.5 * torch.mean(
                    torch.sum(1 + log_var - means ** 2 - log_var.exp(), dim=1),
                    dim=0)
                loss = loss_recon_x + loss_kl

                val_loss += loss.item()
                val_batch_count += 1

            avg_train_loss = train_loss / train_batch_count
            avg_valid_loss = val_loss / val_batch_count

            logger.info(f"Epoch(s): {epoch}\ttrain Loss: {avg_train_loss:.5f}\t"
                        f"validate Loss: {avg_valid_loss:.5f}")

            if early_stopper.early_stop(val_loss):
                logger.info("Early Stopped!")
                break

    def get_anomaly_scores(self, x):
        """Computes the anomaly scores"""
        self.model.eval()

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x.astype(np.float32))

        x_hat, _, _, _ = self.model(x)
        scores = torch.sum(torch.abs(x - x_hat), dim=1).detach().numpy()

        return scores

    def fit(self, x):
        """train the variational auto-encoder model based on
        the latest raw data
        """
        logger.info("Start to execute vae model training...")
        x = x.astype(np.float32)
        x_train, x_val = train_test_split(x, test_size=0.3,
                                          random_state=1234, shuffle=True)

        self.train(x_train, x_val)
        self.model.eval()

        x_tensor = torch.Tensor(x_train)
        x_hat, _, _, _ = self.model(x_tensor)

        scores = torch.sum(torch.abs(x_hat - x_tensor), dim=1).detach().numpy()

        self.post_processes.train(scores)

    def predict(self, x):
        """Predicts the anomaly score by variational auto-encoder model"""
        scores = self.get_anomaly_scores(x)
        scores = self.post_processes(scores)

        return (scores > 0) * 1

    def need_retrain(self, look_back_hours=4, max_error_rate=0.7):
        point_count = look_back_hours * 12
        error_rate = self.post_processes.error_rate(point_count)

        if error_rate > max_error_rate:
            return True

        else:
            return False


class VAE(nn.Module):
    """The variational auto-encoder model implemented by torch"""

    def __init__(self, x_dim, config: VAEConfig):
        """The variational auto-encoder model initializer"""
        super().__init__()

        self.encoder = Encoder(x_dim, config=config)
        self.decoder = Decoder(x_dim, config=config)

    def forward(self, x):
        """The whole pipeline of variational auto-encoder model"""
        means, log_var = self.encoder(x)
        z = self.re_parameterize(means, log_var)
        x_hat = self.decoder(z)

        return x_hat, means, log_var, z

    @staticmethod
    def re_parameterize(means, log_var):
        """Re-parameterize the means and vars"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std


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
        return x_hat


def build_multi_hidden_layers(input_size: int, hidden_sizes: List[int],
                              dropout_rate: float, activation: Callable):
    """build vae model mutli-hidden layers"""
    multi_hidden_layers = []
    for i in range(len(hidden_sizes)):
        in_size = input_size if i == 0 else hidden_sizes[i - 1]
        multi_hidden_layers.append(nn.Linear(in_size, hidden_sizes[i]))
        multi_hidden_layers.append(activation())
        multi_hidden_layers.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*multi_hidden_layers)
