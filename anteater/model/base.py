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
Description: The model base class for config and detector.
"""

import copy
import json
import os
import stat
from os import path

import torch

from anteater.model.post_process.calibrate import Calibrator
from anteater.model.post_process.pipeline import PostProcessPipe
from anteater.model.post_process.threshold import Threshold


class DetectorConfig:
    """The base class for detector configuration"""

    filename = None

    def __init__(
            self,
            max_score=1000,
            alm_threshold=2,
            enable_threshold=True,
            enable_calibrator=True,
            **kwargs):
        """The detector config initializer"""
        self.enable_threshold = enable_threshold
        self.enable_calibrator = enable_calibrator
        self.calibrator = Calibrator(max_score, **kwargs)
        self.threshold = Threshold(alm_threshold, **kwargs)

    def post_processes(self):
        """The post processes of detector"""
        processes = []
        if self.enable_calibrator and self.calibrator is not None:
            processes.append(self.calibrator)
        if self.enable_threshold and self.threshold is not None:
            processes.append(self.threshold)

        return PostProcessPipe(processes)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """class initializer from the config dict"""
        config_dict = dict(**config_dict, **kwargs)
        config = cls(**config_dict)
        cal_config = config_dict.get("calibrator", None)
        thr_config = config_dict.get("threshold", None)
        if cal_config:
            config.calibrator = Calibrator(**cal_config)

        if thr_config:
            config.threshold = Threshold(**thr_config)

        return config

    def to_dict(self):
        """dumps the config dict"""
        config_dict = {}
        for key, val in self.__dict__.items():
            if hasattr(val, "to_dict"):
                val = val.to_dict()

            config_dict[key] = val

        return config_dict


class DetectorBase:
    """The base class for the detector model"""
    filename = None
    config_class = DetectorConfig

    def __init__(self, config):
        """The base class initializer"""
        self.config = config
        self.last_train_time = None

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

    @classmethod
    def load(cls, folder: str, **kwargs):
        """Loads the model from the file"""
        config_file = os.path.join(folder, cls.config_class.filename)
        state_file = os.path.join(folder, cls.filename)

        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(config_file, os.O_RDONLY, modes), "r") as f:
            config_dict = json.load(f)

        config = cls.config_class.from_dict(config_dict, **kwargs)
        model = cls(config=config)

        state_dict = torch.load(state_file)
        if "config" in state_dict:
            state_dict.pop("config")

        for name, val in state_dict.items():
            if hasattr(model, name):
                setattr(model, name, val)

        return model
