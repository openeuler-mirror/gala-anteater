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
from pandas import DataFrame
from scipy.signal import savgol_filter


def is_empty(y):
    """Checks if y is empty or not"""
    if isinstance(y, list) and not y:
        return True

    if isinstance(y, np.ndarray) and not y.any():
        return True

    return False


def conv_smooth(y, box_pts):
    """Apply a convolution smoothing to an array"""
    if is_empty(y):
        return y

    box = np.divide(np.ones(box_pts), box_pts)
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def savgol_smooth(y, window_length, polyorder, *args, **kwargs):
    """Apply a Savitzky-Golay filter to an array"""
    if is_empty(y):
        return y

    return savgol_filter(y, window_length, polyorder, *args, *kwargs)


def smoothing(y, *args, method='conv_smooth', **kwargs):
    """Smooth y base on the smoothing method"""
    if method == 'conv_smooth':
        return conv_smooth(y, *args, **kwargs)
    elif method == 'savgol_smooth':
        return savgol_filter(y, *args, **kwargs)
    else:
        raise ValueError(f'Unknown smoothing method {method}!')


def smooth_data(df: DataFrame, window=3, is_filter_ts: bool = False) -> DataFrame:
    """Smooth the dataframe df by column

    new values is the mean of the rolling window and backfill by
    the next valid observation
    """
    if is_filter_ts:
        for col in df.columns:
            if col == "timestamp":
                continue
            df[col] = df[col].rolling(window=window).mean().bfill().values
    else:
        df = df.rolling(window=window).mean().bfill()

    return df


def moving_average(x: np.ndarray, window: int, stride: int):
    """Sliding window average

    :param x: The data matrix shape=(n,d),
              n is the number of sampling points,
              and d is the data dimension of each point
    :param window: Window length
    :param stride: Window step
    :return: the average result
    """
    n, k = x.shape
    if window > n:
        window = n
    score = np.zeros(n)
    # Record the number of times each point is updated,
    # and represent the weight of the original value of
    # the point as the impact of the new point is added
    score_weight = np.zeros(n)
    # The window starts
    wb = 0
    while True:
        # The window ends
        we = wb + window
        x_window = x[wb:we]
        # The average vector of the window
        x_mean = np.mean(x_window, axis=0)
        # Add influence to the score
        dis = np.sqrt(np.sum(np.square(x_window - x_mean), axis=1))
        score[wb:we] = (score[wb:we] * score_weight[wb:we] + dis)
        score_weight[wb:we] += 1
        score[wb:we] /= score_weight[wb:we]
        if we >= n:
            break
        wb += stride
    # Map score to (0, 1)
    return score


def online_moving_average(
        incoming_data: np.ndarray,
        historical_data: np.ndarray,
        window: int, stride: int):
    """ Online sliding window average
    :param incoming_data: shape=(n,d)
    :param historical_data: shape=(m,d)
    :param window: the window sizes
    :param stride: the window steps
    :return: the moving average result
    """
    n = incoming_data.shape[0]
    # 根据窗口大小计算出所需要的所有数据量(把第一个窗口的终点放在incoming_data的起点)
    need_history_begin = max(0, historical_data.shape[0] - window + 1)
    need_data = np.concatenate(
        [incoming_data, historical_data[need_history_begin:]], axis=0
    )
    score = moving_average(need_data, window, stride)
    # 截取最后n个点的数据表示incoming_data的数值将score映射到(0, 1]上
    return score[0:n]
