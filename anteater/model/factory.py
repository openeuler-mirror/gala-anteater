#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The factory of the model initializer.
"""

import json
import os
import stat

from anteater.model.algorithms.normalization import Normalization
from anteater.model.algorithms.vae import VAEDetector, VAEConfig
from anteater.utils.log import Log

log = Log().get_logger()


class ModelFactory:
    """The model factory"""

    def __init__(self, conf_folder, model_folder, **kwargs):
        """The model factory initializer"""
        self.conf_folder = conf_folder
        self.model_folder = model_folder

    def create(self, name, **kwargs):
        """Creates the model based on model name"""
        if name == "vae":
            try:
                model = VAEDetector.load(self.model_folder)
            except FileNotFoundError:
                log.warning("FileNotFoundError: ModelFactory create 'vae', initializing vae model!")

                modes = stat.S_IWUSR | stat.S_IRUSR
                config_file = os.path.join(self.conf_folder, VAEConfig.filename)
                with os.fdopen(os.open(config_file, os.O_RDONLY, modes), "r") as f:
                    config_dict = json.load(f)

                config = VAEConfig.from_dict(config_dict, **kwargs)
                model = VAEDetector(config)

        elif name == "norm":
            try:
                model = Normalization.load(self.model_folder)
            except FileNotFoundError:
                log.warning("FileNotFoundError: ModelFactory create 'norm', initializing norm model!")
                model = Normalization()

        else:
            raise ValueError(f"Unknown model name {name} when model factorization.")

        return model
