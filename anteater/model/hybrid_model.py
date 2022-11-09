#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The implementation of hybrid model which is a multiple time-series model
"""

import os
import time
from datetime import datetime, timedelta
from functools import reduce
from typing import Tuple, List

import numpy as np
import pandas as pd

from anteater.config import AnteaterConf
from anteater.model.factory import ModelFactory
from anteater.source.metric_loader import MetricLoader
from anteater.utils.data_load import load_metric_operator
from anteater.utils.data_process import parse_operator, metric_value_to_df
from anteater.utils.datetime import datetime_manager
from anteater.utils.log import logger
from anteater.utils.time_series import TimeSeries


class HybridModel:
    """The hybrid model which aims to detect abnormal events
    based on multiple time series dataset.
    """

    def __init__(self, config: AnteaterConf) -> None:
        """The hybrid model initializer"""
        self.config = config
        self.metric_operators = load_metric_operator()
        self.unique_metrics = set([m for m, _ in self.metric_operators])
        self.threshold = config.hybrid_model.threshold
        self.model = config.hybrid_model.name

        self.keep_model = config.hybrid_model.keep_model
        self.model_folder = config.hybrid_model.model_folder
        self.latest_model_folder = config.hybrid_model.latest_model_folder

        self.pipeline = self.select_pipe()

        self._last_retrain = None
        self._min_retrain_hours = 4

    def select_pipe(self):
        factory = ModelFactory(self.model_folder)
        return [
            ("norm", factory.create("norm")),
            ("classifier", factory.create(self.model)),
        ]

    def predict(self, x):
        for pipe in self.pipeline[: -1]:
            x = pipe[1].transform(x)

        y_pred = self.pipeline[-1][1].predict(x)

        return y_pred

    def training(self, x):
        for pipe in self.pipeline[: -1]:
            x = pipe[1].fit_transform(x)

        self.pipeline[-1][1].fit(x)

        if self.keep_model:
            if not os.path.exists(self.latest_model_folder):
                os.makedirs(self.latest_model_folder)

            for pipe in self.pipeline:
                pipe[1].save(self.latest_model_folder)

    def is_abnormal(self, y_pred):
        """Checks if existing abnormal or not"""
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        abnormal = sum(y_pred) >= len(y_pred) * self.threshold

        if abnormal:
            logger.info("Detected anomaly on hybrid model!")

        return abnormal

    def get_training_data(self, start: datetime, end: datetime):
        """Get the training data to support model training"""
        logger.info(f"Get training data during {start} to {end}!")

        _, dfs = self.__get_dataframe(start, end)

        if not dfs:
            return pd.DataFrame()

        x_df = reduce(lambda left, right: pd.concat([left, right], axis=0), dfs)

        return x_df

    def get_inference_data(self) -> Tuple[List[str], List[pd.DataFrame]]:
        """Get data for the model inference and prediction"""
        start, end = datetime_manager.last(minutes=1)
        ids, dfs = self.__get_dataframe(start, end)

        return ids, dfs

    def online_training(self):
        """Checks online training conditions and run online model training"""
        utc_now = datetime_manager.utc_now
        if not self._last_retrain:
            self._last_retrain = utc_now
            return

        if self._last_retrain + timedelta(hours=self._min_retrain_hours) >= utc_now:
            return

        if self.pipeline[-1][1].need_retrain():
            logger.info("Start Online Training!")

            x = self.get_training_data(self._last_retrain, utc_now)
            if x.empty:
                logger.error("Error")
            else:
                logger.info(f"The shape of training data: {x.shape}")
                self.training(x)
                self._last_retrain = utc_now

    def __get_dataframe(self, start: datetime, end: datetime) \
            -> Tuple[List[str], List[pd.DataFrame]]:
        """Gets the features during a period seperated by machine ids"""
        loader = MetricLoader(self.config)

        tim_run = time.time()
        machine_ids = loader.get_unique_label(start, end, self.unique_metrics, label_name="machine_id")

        if machine_ids:
            logger.info(f"Spends: {time.time() - tim_run} seconds to get {len(machine_ids)} unique machine_ids!")
        else:
            logger.warning("Cannot get unique machine ids from PrometheusAdapter!")

        tim_run = time.time()
        dataframes = []
        for machine_id in machine_ids:
            logger.info(f"Fetch metric values from machine: {machine_id}.")

            metric_val_df = []
            for metric, operator in self.metric_operators:
                operator_name, operator_value = parse_operator(operator)
                time_series = loader.get_metric(start, end, metric, label_name="machine_id", label_value=machine_id,
                                                operator_name=operator_name, operator_value=operator_value)

                if len(time_series) > 1:
                    raise ValueError(f'Got multiple time_series based on machine id: {len(time_series)}')

                if time_series:
                    time_series = time_series[0]
                else:
                    time_series = TimeSeries(metric, {}, [], [])

                metric_val_df.append(time_series.to_df())

            df = reduce(lambda left, right: pd.DataFrame(left).join(right, how='outer'), metric_val_df)
            df = df.fillna(0)

            dataframes.append(df)

        if machine_ids:
            logger.info(f"Spends: {time.time() - tim_run} seconds to get get all metric values!")

        return machine_ids, dataframes