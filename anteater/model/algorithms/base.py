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
import os
import stat
from abc import ABC

from anteater.utils.log import logger


class Serializer(ABC):
    """The algorithm post process abstract class"""

    filename = None

    @classmethod
    def load(cls, folder: str, **kwargs):
        """Loads the model from the file"""
        config_file = os.path.join(folder, cls.filename)

        if not os.path.isfile(config_file):
            logger.warning(f'Unknown model file, loads default {cls.__name__} model!')
            return cls(**kwargs)

        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(config_file, os.O_RDONLY, modes), "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        """Loads the object from the dict"""
        config_dict = copy.copy(config_dict)

        return cls(**config_dict)

    def save(self, folder):
        """Saves the model into the file"""
        config_dict = self.to_dict()
        modes = stat.S_IWUSR | stat.S_IRUSR
        config_file = os.path.join(folder, self.filename)
        with os.fdopen(os.open(config_file, os.O_WRONLY | os.O_CREAT, modes), "w") as f:
            f.truncate(0)
            json.dump(config_dict, f, indent=2)

    def to_dict(self):
        """Dumps the object to the dict"""
        state_dict = {}
        for key, val in self.__dict__.items():
            if not key.startswith('_'):
                state_dict[key] = val

        return state_dict
