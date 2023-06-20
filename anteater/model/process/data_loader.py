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
from anteater.model.process.base import PreProcess


class DataLoader:
    """The data loader provides train and test data for the model"""

    def __init__(self, config, is_training=False, **kwargs):
        """The data loader calss initializer"""
        self.config = config

        self.__features = None
        self.__train_times = None
        self.__valid_times = None
        self.__test_times = None
        self.train_np = None
        self.valid_np = None
        self.test_np = None
        self.__scale = None

        self.ori_train = None
        self.ori_valid = None
        self.ori_test = None

        self.preprocessor = PreProcess(self.config)

        self.sli_metrics = kwargs.get("sli_metrics")
        self.machine_id = kwargs.get("machine_id")
        self.detect_metrics = kwargs.get("detect_metrics")

        if is_training:
            self.raw_data = kwargs.get("train_data")
            self.__read_data(self.raw_data)
        else:
            self.__read_test_data(kwargs.get("test_data"))

    @property
    def scale(self):
        """The scale for data scaling"""
        return self.__scale

    @property
    def features(self):
        """The features in model building"""
        return self.__features

    @property
    def train_times(self):
        """The training stage time cost"""
        return self.__train_times

    @property
    def valid_times(self):
        """The validing stage time cost"""
        return self.__valid_times

    @property
    def test_times(self):
        """The testing stage time cost"""
        return self.__test_times

    def return_data(self):
        """The train and valid dataset"""
        return self.train_np, self.valid_np

    def return_test_data(self):
        """The test dataset"""
        return self.test_np

    def __read_data(self, raw_data):
        """Reads training data"""
        channels_pd = raw_data[self.detect_metrics]
        time_stamp_str = raw_data["timestamp"]
        channels_pd['timestamp'] = pd.to_datetime(time_stamp_str).values

        train_df, valid_df = self.preprocessor.split_data(channels_pd)

        self.__train_times, self.__valid_times = train_df.timestamp.astype(
            str).tolist(), valid_df.timestamp.astype(str).tolist()
        self.__features = np.array(channels_pd.columns)[:-1]
        self.train_np, self.valid_np = \
            train_df.values[:, :-1], valid_df.values[:, :-1]

        self.train_np = self.preprocessor.fit_transform(self.train_np)
        self.valid_np = self.preprocessor.transform(self.valid_np)

    def __read_test_data(self, df):
        """Reads test data"""
        channels_pd = df[self.detect_metrics]
        time_stamp_str = df["timestamp"]
        channels_pd['timestamp'] = pd.to_datetime(time_stamp_str).values

        test_df = channels_pd
        self.__test_times = test_df.timestamp.astype(str).tolist()
        self.__features = np.array(channels_pd.columns)[:-1]
        self.test_np = test_df.values[:, :-1]
        self.ori_test = self.test_np
        self.test_np = self.preprocessor.transform(self.valid_np)
