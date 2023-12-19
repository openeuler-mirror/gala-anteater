import os
import pickle
import stat

import numpy as np
import pandas as pd
from anteater.model.algorithms.usad_pre_processor import PreProcessor
from anteater.utils.log import logger


def smooth_data(df, window=3):
    """Smooths metrics"""
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = df[col].rolling(window=window).mean().bfill().values
    return df


class UsadDataLoader:
    def __init__(self, config, is_training=False, **kwargs):
        self.config = config

        self.__features = None
        self.__train_times = None
        self.__valid_times = None
        self.__test_times = None
        self.train_np = None
        self.valid_np = None
        self.test_np = None

        self.ori_train = None
        self.ori_valid = None
        self.ori_test = None

        self.sli_metrics = kwargs.get("sli_metrics")
        self.machine_id = kwargs.get("machine_id")
        self.detect_metrics = kwargs.get("detect_metrics")
        self.__scale = kwargs.get("scale", None)

        if is_training:
            self.raw_data = kwargs.get("train_data")
            self.__read_data(self.raw_data)
        else:
            self.__read_test_data(kwargs.get("test_data"))

    @property
    def scale(self):
        return self.__scale

    @property
    def features(self):
        return self.__features

    @property
    def train_times(self):
        return self.__train_times

    @property
    def valid_times(self):
        return self.__valid_times

    @property
    def test_times(self):
        return self.__test_times

    def return_data(self):
        return self.train_np, self.valid_np

    def return_test_data(self):
        return self.test_np

    def __read_data(self, raw_data):
        """Reads training data"""
        metric_raw = raw_data
        metric_candidates = [col for col in metric_raw.columns]

        metric = smooth_data(
            metric_raw, window=self.config.params["usad"]["smooth_window"])
        channels_pd = metric[self.detect_metrics]
        time_stamp_str = metric["timestamp"]
        channels_pd['timestamp'] = pd.to_datetime(time_stamp_str).values

        train_df, valid_df = self.__split_data(channels_pd)

        self.__train_times, self.__valid_times = train_df.timestamp.astype(
            str).tolist(), valid_df.timestamp.astype(str).tolist()
        self.__features = np.array(channels_pd.columns)[:-1]
        self.train_np, self.valid_np = train_df.values[:,
                                                       :-1], valid_df.values[:, :-1]
        self.train_np, self.valid_np, self.__scale, self.ori_train, self.ori_valid = PreProcessor.preprocess(
            self.train_np,
            self.valid_np,
            self.config.params["usad"]["preprocess_type"],
            clip_alpha=self.config.params["usad"]["clip_alpha"])

    def __read_test_data(self, df):
        """Reads test data"""
        metric_raw = df
        metric = metric_raw
        metric = smooth_data(
            metric_raw, window=self.config.params["usad"]["smooth_window"])

        channels_pd = metric[self.detect_metrics]
        time_stamp_str = metric["timestamp"]
        channels_pd['timestamp'] = pd.to_datetime(time_stamp_str).values

        test_df = channels_pd
        self.__test_times = test_df.timestamp.astype(str).tolist()
        self.__features = np.array(channels_pd.columns)[:-1]
        self.test_np = test_df.values[:, :-1]

        # raw
        self.test_np, self.ori_test = PreProcessor.test_preprocess(
            self.__scale,
            self.test_np,
            self.config.params["usad"]["preprocess_type"],
            clip_alpha=self.config.params["usad"]["clip_alpha"])

    def __split_data(self, rawdata_df):
        """"Divides the training and validation data"""
        n = int(len(rawdata_df) * 0.1)
        train_df, valid_df = rawdata_df[:-n], rawdata_df[-n:]

        return train_df, valid_df
