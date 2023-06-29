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

import logging

import numpy as np

from anteater.core.kpi import ModelConfig
from anteater.model.factory import ModelFactory as factory
from anteater.model.process.base import PreProcess
from anteater.model.process.data_loader import TrainTestDataLoader
from anteater.model.process.post_process import PostProcessor
from anteater.utils.constants import POINTS_MINUTE


class OnlineUsadModel:
    """Online Usad model"""

    def __init__(self, config: ModelConfig) -> None:
        """The model initializer"""
        params = config.params
        self.config = config
        self.folder = config.model_path

        self.preprocessor = PreProcess(self.config)
        self.model = factory.create_model('usad', self.folder, **params)

        self.params = config.params
        self.th = self.params.get('th')

        self.post = PostProcessor(config)

    def predict(self, x, machine_id, detect_metrics, sli_metrics):
        """Runs online model predicting"""
        data_loader = TrainTestDataLoader(self.config, is_training=False, machine_id=machine_id,
                                          test_data=x, detect_metrics=detect_metrics, sli_metrics=sli_metrics)
        x_test = data_loader.return_test_data()
        x_g, x_g_d = self.model.predict(x_test)
        scores = self.post.compute_score(x_test, x_g, x_g_d)
        thresholds = self.post.spot_run(scores)
        health_anomalies = self.post.health_detection(x_test, scores, thresholds)
        y_pred = self.post.get_anomalies(x_test, health_anomalies)
        return y_pred

    def train(self, df, machine_id, detect_metrics, sli_metrics):
        """Runs online model training"""
        data_loader = TrainTestDataLoader(self.config, is_training=True, machine_id=machine_id,
                                          train_data=df, detect_metrics=detect_metrics, sli_metrics=sli_metrics)
        x_train, x_valid = data_loader.return_data()
        self.model.train(x_train, x_valid)

    def is_abnormal(self, y_pred):
        """Checks if existing abnormal or not"""
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        if len(y_pred) > POINTS_MINUTE:
            y_pred = y_pred[-POINTS_MINUTE:]

        if len(y_pred) < POINTS_MINUTE:
            logging.warning(
                f'The length of y_pred is less than {POINTS_MINUTE}')
            return False

        abnormal = sum([1 for y in y_pred if y > 0]) >= len(y_pred) * self.th

        if abnormal:
            logging.info(
                f'Detects abnormal events by {self.__class__.__name__}!')

        return abnormal
