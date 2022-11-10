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
"""
Time:
Author:
Description: The factory of the model initializer.
"""

from anteater.model.algorithms.normalization import Normalization
from anteater.model.algorithms.vae import VAEDetector, VAEConfig
from anteater.utils.log import logger


class ModelFactory:
    """The model factory"""

    def __init__(self, model_folder, **kwargs):
        """The model factory initializer"""
        self.model_folder = model_folder

    def create(self, name, **kwargs):
        """Creates the model based on model name"""
        if name == "vae":
            try:
                model = VAEDetector.load(self.model_folder)
            except FileNotFoundError:
                logger.warning("Initializing default vae model!")
                model = VAEDetector(VAEConfig())

        elif name == "norm":
            try:
                model = Normalization.load(self.model_folder)
            except FileNotFoundError:
                logger.warning("Initializing default norm model!")
                model = Normalization()

        else:
            raise ValueError(f"Unknown model name {name} when model factorization.")

        return model
