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
Description: The implementation of early stopping during the model training progress
"""

import numpy as np


class EarlyStopper:
    """The implementation of early stop algorithm for neural network optimizer"""

    def __init__(self, patience=5):
        """The early stopper initializer"""
        self.patience = patience
        self.counter = 0

        self._min_val_loss = np.inf

    def early_stop(self, val_loss, *args, **kwargs):
        """The callable object"""
        if val_loss < self._min_val_loss:
            self._min_val_loss = val_loss
            self.counter = 0

        if val_loss > self._min_val_loss:
            self.counter += 1

        if self.counter > self.patience:
            return True

        return False
