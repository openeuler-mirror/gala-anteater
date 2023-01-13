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

from typing import List

import numpy as np

from anteater.core.anomaly import AnomalyTrend
from anteater.model.algorithms.smooth import conv_smooth
from anteater.utils.common import divide


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
    y = conv_smooth(y, box_pts=7)

    if not win_len:
        win_len = len(y) // 2

    if divide(np.mean(y[:win_len]), np.mean(y[-win_len:])) < 0.9:
        return 1

    elif divide(np.mean(y[:win_len]), np.mean(y[-win_len:])) > 1.1:
        return -1

    else:
        return 0


def check_trend(values: List[float], atrend: AnomalyTrend):
    """Checks the values with an 'atrend' trend"""
    if atrend == AnomalyTrend.RISE and trend(values) != 1:
        return False

    if atrend == AnomalyTrend.FALL and trend(values) != -1:
        return False

    return True
