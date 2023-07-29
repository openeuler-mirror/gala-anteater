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

import os
from functools import partial
from typing import Union, Tuple, Callable, Dict

import joblib
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from anteater.model.algorithms.scaler import ClipScaler
from anteater.model.algorithms.smooth import conv_smooth, \
    savgol_smooth, smooth_data


class PreProcessor:
    """The preprocessor base class

    which provides raw data smoothing, scaling, and splitting
    """

    filename = 'preprocessor.pkl'

    def __init__(self, params: Dict) -> None:
        """The preprocessor class initializer"""
        self.params = params.get('preprocessor')
        self.smoothing = self.__select_smoothing()
        self.scaler = self.__select_scaler()

    @classmethod
    def load(cls, folder: str, **kwargs):
        """Load model from the file"""
        file = os.path.join(folder, cls.filename)
        model = cls(**kwargs)
        model.scaler = joblib.load(file)

        return model

    def save(self, folder: str):
        """Save the scaler into the file"""
        model_path = os.path.join(folder, self.filename)
        joblib.dump(self.scaler, model_path)

    def split_data(self, x: DataFrame, default_size: float = 0.7) -> Tuple[DataFrame, DataFrame]:
        """"Divides the data x into train and valid dataset"""
        train_size = self.params.get('train_size')
        valid_size = self.params.get('valid_size')
        shuffle = self.params.get('shuffle')

        if not train_size and not valid_size:
            train_size, valid_size = default_size, 1 - default_size

        if shuffle is None:
            shuffle = False

        if isinstance(x, np.ndarray):
            x_train, x_valid = train_test_split(
                x, test_size=valid_size, shuffle=shuffle)
        elif isinstance(x, DataFrame):
            n = round(x.shape[0] * train_size)
            x_train = x[:n]
            x_valid = x[n:]
        else:
            raise TypeError(f'Unknown X type: {type(x)}')

        return x_train, x_valid

    def fit(self, x: DataFrame):
        """Compute the arguments of the scaler"""
        x = self.smoothing(x)
        self.scaler.fit_transform(x)

        return self

    def fit_transform(self, x: DataFrame):
        """Compute arguments, then mooting and scaling the features"""
        x = self.smoothing(x)
        x = self.scaler.fit_transform(x)

        return x

    def transform(self, x: DataFrame):
        """Scale the features based on the data"""
        x = self.smoothing(x)
        x = self.scaler.transform(x)

        return x

    def __select_scaler(self) -> Union[MinMaxScaler, StandardScaler, ClipScaler]:
        """Select scaler based on scaler type"""
        scale_type = self.params.get('scale_type')
        if scale_type == 'minmax':
            scaler = MinMaxScaler()
        elif scale_type == 'standard':
            scaler = StandardScaler()
        elif scale_type == 'clip':
            alpha = self.params.get('clip_alpha')
            scaler = ClipScaler(alpha=alpha)
        else:
            raise ValueError(f'Unknown scaler type: {scale_type}')

        return scaler

    def __select_smoothing(self) -> Callable:
        """Select smoother method based on smooth type"""
        smooth_type = self.params.get('smooth_type')
        if smooth_type == 'rolling':
            window = self.params.get('smooth_window')
            smoothing = partial(smooth_data, window=window)

        elif smooth_type == 'conv_smooth':
            box_pts = self.params.get('box_pts')
            smoothing = partial(conv_smooth, box_pts=box_pts)

        elif smooth_type == 'savgol_smooth':
            window_length = self.params.get('window_length')
            polyorder = self.params.get('polyorder')
            smoothing = partial(savgol_smooth,
                                window_length=window_length,
                                polyorder=polyorder)

        else:
            raise ValueError(f'Unknown smoothing type: {smooth_type}')

        return smoothing
