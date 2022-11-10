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
Description: The base class for post-process of the algorithms
"""

import copy
from abc import ABC


class PostProcess(ABC):
    """Post process base class"""

    def __call__(self, scores):
        """The callable object"""
        raise NotImplementedError

    def train(self, scores):
        """Trans the model"""
        raise NotImplementedError

    def to_dict(self):
        """Dumps the object to the dict"""
        state_dict = {}
        for key, val in self.__dict__.items():
            if not key.startswith('_'):
                state_dict[key] = val

        state_dict["name"] = type(self).__name__

        return state_dict

    @classmethod
    def from_dict(cls, state_dict):
        """Loads the object from the dict"""
        state_dict = copy.copy(state_dict)
        state_dict.pop("name", None)

        return cls(**state_dict)
