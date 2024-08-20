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


def n_sigma(values, obs_size, n=3, method="abs"):
    """The 'N-sigma rule' outlier detect function"""
    if obs_size <= 0:
        raise ValueError("The obs_size should great than zero!")
    if len(values) <= obs_size:
        raise ValueError("The obs_size should be great than values length")
    train_val = values[:-obs_size]
    obs_val = values[-obs_size:]

    mean = np.mean(train_val)
    std = np.std(train_val)

    if method == "abs":
        outlier = [val for val in obs_val if abs(val - mean) > n * std]
    elif method == 'min':
        outlier = [val for val in obs_val if val < mean - n * std]
    elif method == 'max':
        outlier = [val for val in obs_val if val > mean + n * std]
    else:
        raise ValueError(f'Unknown method {method}')

    return outlier, mean, std


def n_sigma_ex(train_val, values, obs_size, n=3, method="abs"):
    """The 'N-sigma rule' outlier detect function"""
    if obs_size <= 0:
        raise ValueError("The obs_size should great than zero!")
    if len(values) <= obs_size:
        raise ValueError("The obs_size should be great than values length")
    obs_val = values[-obs_size:]
    mean = np.mean(train_val)
    std = np.std(train_val)
    if method == "abs":
        outlier = [val for val in obs_val if abs(val - mean) > n * std]
        outlier_idx = np.array(abs(obs_val - mean) > n * std, dtype=np.int32)
    elif method == 'min':
        outlier = [val for val in obs_val if val < mean - n * std]
        outlier_idx = np.array(obs_val < mean - n * std, dtype=np.int32)
    elif method == 'max':
        outlier = [val for val in obs_val if val > mean + n * std]
        outlier_idx = np.array(obs_val > mean + n * std, dtype=np.int32)
    else:
        raise ValueError(f'Unknown method {method}')

    return outlier, outlier_idx, mean, std
