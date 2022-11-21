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

from anteater.model.smoother import conv_smooth


def slope(y, win_len):
    """Calculates point slope in an array"""
    if len(y) <= win_len:
        raise ValueError('point_slope: the length of array should'
                         f'greater than window_length : f{win_len}.')

    return np.divide(np.subtract(y[win_len:], y[: -win_len]), y[: -win_len])


def smooth_slope(time_series, windows_length):
    val = conv_smooth(time_series.to_df(), box_pts=13)
    val = slope(val, win_len=13)
    return val[-windows_length:]


def trend(y, win_len=None):
    """Gets the trend for the y"""
    if not win_len:
        win_len = len(y) // 2

    if np.mean(y[:win_len]) < np.mean(y[-win_len:]):
        return 1
    
    elif np.mean(y[:win_len]) > np.mean(y[-win_len:]):
        return -1
    
    else:
        return 0
