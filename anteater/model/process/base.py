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

from functools import partial
import os
from typing import Union, Tuple
import joblib
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from anteater.model.algorithms.scaler import ClipScaler
from anteater.model.algorithms.smooth import conv_smooth, \
    savgol_smooth, smooth_data


class PreProcess:
    """The preprocess base class

    which provides raw data smoting, scaling, and spliting
    """

    filename = 'preprocess.pkl'

    def __init__(self, config, *args, **kwargs) -> None:
        """The preprocess class initializer"""
        self.config = config.get('preprocess')

        self.smooth = self._select_smoother()
        self.scaler = self._select_scaler()

    @classmethod
    def load(cls, folder, **kwargs):
        """Load model from the file"""
        file = os.path.join(folder, cls.filename)
        model = cls(**kwargs)
        model.scaler = joblib.load(file)
        return model

    def save(self, folder):
        """Save the scaler into the file"""
        model_path = os.path.join(folder, self.filename)
        joblib.dump(self.scaler, model_path)

    def split_data(self, x, default_size=0.7) -> Tuple[DataFrame, DataFrame]:
        """"Divides the train and valid data"""
        train_size = self.config.get('train_size')
        valid_size = self.config.get('valid_size')
        shuffle = self.config.get('shuffle')

        if not train_size and not valid_size:
            train_size, valid_size = default_size, 1 - default_size

        if shuffle is None:
            shuffle = False

        if isinstance(x, np.ndarray):
            x_train, x_valid = train_test_split(
                x, test_size=valid_size, shuffle=shuffle)
        elif isinstance(x, DataFrame):
            n = len(x) * train_size
            x_train = x[:-n]
            x_valid = x[-n:]
        else:
            raise TypeError(f'Unknown X type: {type(x)}')

        return x_train, x_valid

    def fit(self, x):
        """Compute the arguments of the scaler"""
        x = self.smooth(x)
        self.scaler.fit_transform(x)

        return self

    def fit_transform(self, x):
        """Compute arguments, then mooting and scaling the features"""
        x = self.smooth(x)
        x = self.scaler.fit_transform(x)

        return x

    def transform(self, x):
        """Scale the features based on the data"""
        x = self.smooth(x)
        x = self.scaler.fit_transform(x)

        return x

    def _select_scaler(self) -> \
            Union[MinMaxScaler, StandardScaler, ClipScaler]:
        """Select scaler based on scaler type"""
        scl_type = self.config.get('scale_type')
        if scl_type == 'minmax':
            scaler = MinMaxScaler()
        elif scl_type == 'standard':
            scaler = StandardScaler()
        elif scl_type == 'clip':
            alpha = self.config.get('clip_alpha')
            scaler = ClipScaler(alpha=alpha)
        else:
            raise ValueError(f'Unknown scaler type: {scl_type}')

        return scaler

    def _select_smoother(self) -> partial:
        """Select smoother method based on smooth type"""
        smt_type = self.config('smooth_type')
        if smt_type == 'rolling':
            window = self.config['window']
            smooth = partial(smooth_data, window=window)
        elif smt_type == 'conv_smooth':
            box_pts = self.config['box_pts']
            smooth = partial(conv_smooth, box_pts=box_pts)
        elif smt_type == 'savgol_smooth':
            window_length = self.config['window_length']
            polyorder = self.config['polyorder']
            smooth = partial(savgol_smooth,
                             window_length=window_length,
                             polyorder=polyorder)
        else:
            raise ValueError(f'Unknown smooth type: {smt_type}')

        return smooth
