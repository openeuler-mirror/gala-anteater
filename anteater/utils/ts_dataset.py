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

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TSDataset(Dataset):
    """The override of torch dataset class"""
    def __init__(self, x, win_size, step_size):
        self.x = x

        if win_size < step_size:
            raise ValueError("The win_size should great or equal than the step_size!")

        self.win_size = win_size
        self.step_size = step_size
        self.num_samples = (x.shape[0] - win_size) // step_size + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.num_samples + idx

        i = idx * self.step_size
        if isinstance(self.x, np.ndarray):
            return self.x[i: i+self.win_size, :]

        elif isinstance(self.x, pd.DataFrame):
            item = self.x.iloc[i: i + self.win_size, :].values
            item = item.astype(np.float32)
            return item

        else:
            raise TypeError(f'__getitem__ failed, data type: {type(self.x)}')
