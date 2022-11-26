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
from scipy.signal import savgol_filter


def conv_smooth(y, box_pts):
    """Apply a convolution smoothing to an array"""
    box = np.divide(np.ones(box_pts), box_pts)
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def savgol_smooth(y, window_length, polyorder, *args, **kwargs):
    """Apply a Savitzky-Golay filter to an array"""
    return savgol_filter(y, window_length, polyorder, *args, *kwargs)


def smoothing(y, method='conv_smooth', *args, **kwargs):
    if method == 'conv_smooth':
        return conv_smooth(y, *args, **kwargs)
    elif method == 'savgol_smooth':
        return savgol_filter(y, *args, **kwargs)
    else:
        raise ValueError(f'Unknown smoothing method {method}!')

