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

import os
import numpy as np

from typing import Dict
from datetime import datetime, timezone

from anteater.core.kpi import ModelConfig
from anteater.model.algorithms.usad import USADModel
from anteater.model.process.usad_data_loader import UsadDataLoader
from anteater.model.factory import ModelFactory as factory
from anteater.model.process.post_process import PostProcessor
from anteater.utils.constants import POINTS_MINUTE
from anteater.utils.log import logger


class OnlineUsadModel:
    """Online Usad model"""

    def __init__(self, config: ModelConfig) -> None:
        """The model initializer"""
        self.config = config
        self.params = config.params

        self.models: Dict[str, USADModel] = {}
        self.models_info = {}
        self.all_features = None

        self.post = None
        self.metrics_to_delete = []
        self.sli_metrics = None

    def get_min_predict_minutes(self):
        """Gets minimal minutes for model prediction"""
        return self.params.get('min_predict_minutes')

    def get_min_training_hours(self):
        """Gets minimal minutes for model training"""
        return self.params.get('min_train_hours')

    def check_processed_data(self, test_processed):
        for i in range(len(self.all_features)):
            if self.all_features[i] in self.sli_metrics:
                continue
            if np.all(test_processed[:, i] == test_processed[:, i][0]):
                self.metrics_to_delete.append(self.all_features[i])

    @staticmethod
    def concate_data(model_info, test_predict_g, test_predict_g_d, test_processed, test_times):
        train_predict_g_d, valid_predict_g_d = model_info['train_predict_g_d'], model_info['valid_predict_g_d']
        train_predict_g, valid_predict_g = model_info['train_predict_g'], model_info['valid_predict_g']

        x_all_processed = np.concatenate((model_info['train_processed'][-len(train_predict_g_d):],
                                          model_info['valid_processed'][-len(valid_predict_g_d):],
                                          test_processed[-len(test_predict_g_d):]), axis=0)

        all_g = np.concatenate((train_predict_g, valid_predict_g, test_predict_g), axis=0)
        all_g_d = np.concatenate((train_predict_g_d, valid_predict_g_d, test_predict_g_d), axis=0)
        all_times = np.concatenate((model_info['train_times'][-len(train_predict_g_d):],
                                    model_info['valid_times'][-len(valid_predict_g_d):],
                                    test_times[-len(test_predict_g_d):]), axis=0)

        return x_all_processed, all_g, all_g_d, all_times

    def predict(self, last_5_min_x_dfs, df, machine_id, detect_metrics, sli_metrics):
        """Runs online model predicting"""
        is_training = False
        self.sli_metrics = sli_metrics
        model_info = self.models_info[machine_id]

        # test_data
        test_data_loader_ = UsadDataLoader(self.config, is_training=is_training, machine_id=machine_id,
                                       test_data=last_5_min_x_dfs, detect_metrics=detect_metrics,
                                       sli_metrics=sli_metrics, scale=model_info['scale'])
        last_5_min_test_processed = test_data_loader_.return_test_data()
        # check process or delete metric
        self.all_features = test_data_loader_.features
        self.check_processed_data(last_5_min_test_processed)

        test_data_loader = UsadDataLoader(self.config, is_training=is_training, machine_id=machine_id,
                                      test_data=df, detect_metrics=detect_metrics, sli_metrics=sli_metrics,
                                      scale=model_info['scale'])
        test_processed = test_data_loader.return_test_data()
        ori_test = test_data_loader.ori_test
        test_times = test_data_loader.test_times

        model = self.models[machine_id]
        test_predict_g, test_predict_g_d = model.predict(test_processed)

        train_predict_g_d, valid_predict_g_d = model_info['train_predict_g_d'], model_info['valid_predict_g_d']
        x_all_processed, all_g, all_g_d, all_times = self.concate_data(model_info, test_predict_g, test_predict_g_d,
                                                                       test_processed, test_times)
        # used for cal spot thr.
        self.post = PostProcessor(config=self.config,
                                  machine_id=machine_id,
                                  processed_data=x_all_processed,
                                  reconstruct_data=all_g_d,
                                  generate_data=all_g,
                                  data_info=model_info['data_info'],
                                  all_features=self.all_features.tolist(),
                                  all_times=all_times,
                                  scaler=model_info['scale'],
                                  ori_train=model_info['ori_train'][-len(train_predict_g_d):],
                                  ori_valid=model_info['ori_valid'][-len(valid_predict_g_d):],
                                  ori_test=ori_test[-len(test_predict_g_d):],
                                  sli_metrics=sli_metrics
                                  )

        abnormal_sli_metric = self.post.compute_score(df, machine_id, metrics_to_delete=self.metrics_to_delete)

        spot_detect_res, sli_spot_detect_res = self.post.spot_fit()
        self.post.health_detection()

        y_pred, sli_pred = self.post.get_anomalies()

        return y_pred, sli_pred, abnormal_sli_metric

    def trigger_locate(self, pearson_record):
        """Gets the possible root cause metrics of the anomaly"""
        now_time = datetime.now(timezone.utc).astimezone().astimezone()
        cause_list, anomaly_scores_list = self.post.locate_cause(self.metrics_to_delete, pearson_record)

        return cause_list, now_time, anomaly_scores_list

    def train(self, df, machine_id, detect_metrics, sli_metrics):
        """Runs online model training"""
        is_training = True
        sub_directory = os.path.join(self.config.model_path, machine_id)
        model = factory.create_model('usad', sub_directory, **self.params)

        train_data_loader = UsadDataLoader(self.config, is_training=is_training, machine_id=machine_id,
                                       train_data=df, detect_metrics=detect_metrics, sli_metrics=sli_metrics)
        train_processed, valid_processed = train_data_loader.return_data()

        model.train(train_processed, valid_processed)

        # spot fit
        train_predict_g, train_predict_g_d = model.predict(train_processed)
        valid_predict_g, valid_predict_g_d = model.predict(valid_processed)

        data_info = {
            'train_length': train_predict_g_d.shape[0],
            'valid_length': valid_predict_g_d.shape[0],
            'dataset': machine_id
        }

        # save two components, model | post
        self.models[machine_id] = model
        save_model_info = {
            'scale': train_data_loader.scale,
            'ori_train': train_data_loader.ori_train,
            'ori_valid': train_data_loader.ori_valid,
            'train_times': train_data_loader.train_times,
            'valid_times': train_data_loader.valid_times,
            'train_processed': train_processed,
            'valid_processed': valid_processed,
            'train_predict_g': train_predict_g,
            'train_predict_g_d': train_predict_g_d,
            'valid_predict_g': valid_predict_g,
            'valid_predict_g_d': valid_predict_g_d,
            'data_info': data_info,
        }
        self.models_info[machine_id] = save_model_info

    def is_abnormal(self, y_pred, machine_id):
        """Checks if existing abnormal or not"""
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        if len(y_pred) > POINTS_MINUTE:
            y_pred = y_pred[-POINTS_MINUTE:]

        if len(y_pred) < POINTS_MINUTE:
            logger.warning('The length of y_pred is less than %d',
                           POINTS_MINUTE)
            return False

        th = self.params.get('th')
        anomaly_len = sum([1 for y in y_pred if y > 0])
        test_len = len(y_pred)
        abnormal = anomaly_len >= test_len * th
        logger.info(f'Cur machine: {machine_id}, is_abnormal: {anomaly_len}, {test_len}, {th}')

        if abnormal:
            logger.info('Detects abnormal events by %s!', {self.__class__.__name__})

        return abnormal

    def need_training(self, machine_id):
        """Checks model need to be training before predicting"""
        if machine_id not in self.models:
            return True

        return False
