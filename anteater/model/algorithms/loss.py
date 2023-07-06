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

import numpy as np
import pandas as pd
from scipy import integrate


def point_wise_loss(y, y_hat):
    """Computes point wise loss which tasks the absolute loss
    between each index value in target 'y' and it's corresponding
    index values in predicted 'y_hat'.
    """

    return np.sum(np.abs(y - y_hat), axis=1)


def area_loss(y, y_hat, window=10):
    """Computes the area loss which is created using a fixed length
    windows size that measures the similarity between smoothed local regions
    """

    smooth_y = pd.Series(y).rolling(
        window, center=True, min_periods=window // 2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(
        window, center=True, min_periods=window // 2).apply(integrate.trapz)

    errors = np.sum(np.abs(smooth_y - smooth_y_hat), axis=1)

    return errors


def dtw_loss(y, y_hat, window=10):
    """Computes dynamic time warping(dtw) loss which measures the best similarity
    of all point wise mappings from target series to predicted series, no matter
    how long or difference of series.
    """

    return NotImplemented("'dtw_loss' is not implemented!")


def gan_loss(y, y_g, y_g_d, loss_type, alpha):
    """Compute GAN based model loss based on original data y,
    generated data y_g, and generated and decoded data y_g_d
    """

    if loss_type == 'common':
        loss = alpha * np.square(y - y_g) + (1 - alpha) * np.square(y - y_g_d)

    elif loss_type == 'percentage':
        scale = np.array([1] * y.shape[1])
        loss = np.divide(np.abs(np.subtract(y, y_g_d)), scale)

    elif loss_type == 'power10':
        scale = np.array([1] * y.shape[1])
        power_y = np.power(10, np.divide(y, scale))
        power_y_g_d = np.power(10, np.divide(y_g_d, scale))
        loss = np.subtract(power_y, power_y_g_d)
        loss = np.abs(loss)

    else:
        raise NotImplementedError(f'Unknown loss type: {loss_type}!')

    return loss
