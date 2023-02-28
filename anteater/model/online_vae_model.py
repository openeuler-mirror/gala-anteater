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

from datetime import timedelta

import numpy as np

from anteater.core.kpi import ModelConfig
from anteater.model.factory import ModelFactory as factory
from anteater.utils.constants import POINTS_MINUTE
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class OnlineVAEModel:
    """Online vae model"""
    def __init__(self, config: ModelConfig) -> None:
        """The hybrid model initializer"""
        params = config.params
        self.th = params.get('th')
        self.max_error_rate = params.get('max_error_rate')
        self.min_retrain_hours = params.get('min_retrain_hours')
        self.min_predict_minutes = params.get('min_predict_minutes')

        self.folder = config.model_path
        self.norm = factory.create_model('norm', self.folder, **params)
        self.vae = factory.create_model('vae', self.folder, **params)
        self.calibrate = factory.create_model('error_calibrate', self.folder, **params)
        self.threshold = factory.create_model('dy_threshold', self.folder, **params)

        self.__last_retrain = None

    def validate_x(self, x):
        """Validates the x efficiency, including length, etc"""
        if x.shape[0] < self.min_predict_minutes * 12:
            return False

        return True

    def predict(self, x):
        if not self.validate_x(x):
            return []

        norm_x = self.norm.fit_transform(x)
        anomaly_scores = self.vae.predict(norm_x)
        calibrated_scores = self.calibrate(anomaly_scores)
        y_pred = self.threshold(calibrated_scores)

        return y_pred

    def training(self, x):
        norm_x = self.norm.fit_transform(x)
        anomaly_scores = self.vae.fit_transform(norm_x)
        calibrated_scores = self.calibrate.fit_transform(anomaly_scores)
        self.threshold.fit(calibrated_scores)
        self.save(self.folder)

    def is_abnormal(self, y_pred):
        """Checks if existing abnormal or not"""
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        if len(y_pred) > POINTS_MINUTE:
            y_pred = y_pred[-POINTS_MINUTE:]

        if len(y_pred) < POINTS_MINUTE:
            logger.warning(f'The length of y_pred is less than {POINTS_MINUTE}')
            return False

        abnormal = sum([1 for y in y_pred if y > 0]) >= len(y_pred) * self.th

        if abnormal:
            logger.info(f'Detects abnormal events by {self.__class__.__name__}!')

        return abnormal

    def need_retrain(self, look_back_hours=6):
        """Checks need retrain model or not"""
        utc_now = dt.utc_now()

        if not self.vae.model:
            return True

        if not self.__last_retrain:
            self.__last_retrain = utc_now
            return False

        if self.__last_retrain + timedelta(hours=self.min_retrain_hours) >= utc_now:
            return False

        else:
            self.__last_retrain = utc_now
            return True

    def online_training(self, x):
        """Executes online training and run online model training"""
        logger.info("Start Online Training!")
        if not x:
            logger.error("Error: Empty input x")
        else:
            logger.info(f"The shape of training data: {x.shape}")
            self.training(x)
            self.__last_retrain = dt.utc_now()

    def save(self, folder):
        """Save the model to the specified folder"""
        self.norm.save(folder)
        self.vae.save(folder)
        self.calibrate.save(folder)
        self.threshold.save(folder)
