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

import json
import os
import stat
from datetime import datetime, timezone

import numpy as np
from anteater.core.kpi import ModelConfig
from anteater.model.algorithms.online_ad import OnlineAd
from anteater.model.algorithms.online_ad_helper import DataLoader
from anteater.model.algorithms.online_ad_helper import PostProcessor
from anteater.utils.constants import POINTS_MINUTE
from anteater.utils.log import logger


class OnlineModel:
    """Online model"""

    def __init__(self, config: ModelConfig) -> None:
        """The model initializer"""
        self.config = config
        self.params = config.params
        self.th = self.params.get('th')
        self.max_error_rate = self.params.get('max_error_rate')
        self.min_retrain_hours = self.params.get('min_retrain_hours')
        self.min_predict_minutes = self.params.get('min_predict_minutes')
        self.train_processed = None
        self.valid_processed = None
        self.train_times = None
        self.valid_times = None
        self.ori_train = None
        self.ori_valid = None
        self.train_predict_g = None
        self.train_predict_g_d = None
        self.valid_predict_g = None
        self.valid_predict_g_d = None
        self.data_info = None
        self.test_predict_g = None
        self.test_predict_g_d = None
        self.x_all_processed = None
        self.all_g = None
        self.all_g_d = None
        self.all_times = None
        self.test_processed = None
        self.all_features = None
        self.scale = None
        self.ori_test = None
        self.machine_id = None
        self.post = None
        self.model = None

    def get_machine_id(self, machine_id):
        self.machine_id = machine_id

    def validate_x(self, x):
        """Validates the x efficiency, including length, etc"""
        if x.shape[0] < self.min_predict_minutes * 12:
            return False

        return True

    def restore_model(self):
        """Loads the model"""
        middle_path = os.path.join(
            self.params["output_params"]["middle_dir"], self.machine_id)
        model_path = os.path.join(
            self.params["output_params"]["model_dir"], self.machine_id)
        if not os.path.exists(middle_path) or not os.path.exists(model_path):
            os.makedirs(middle_path)
            os.makedirs(model_path)
            os.makedirs(os.path.join(
                self.params["output_params"]["result_dir"], self.machine_id))
            os.makedirs(os.path.join(
                self.params["input_params"]["training_data_dir"], self.machine_id))
            return True
        if not os.path.exists(os.path.join(
                middle_path, self.params["output_params"]["train_predict_G_D"])):
            return True
        self.train_processed = np.load(os.path.join(
            middle_path, self.params["output_params"]["train_processed"]), allow_pickle=True)
        self.valid_processed = np.load(os.path.join(
            middle_path, self.params["output_params"]["valid_processed"]), allow_pickle=True)

        self.train_times = np.load(os.path.join(
            middle_path, self.params["output_params"]["train_times"]), allow_pickle=True)
        self.valid_times = np.load(os.path.join(
            middle_path, self.params["output_params"]["valid_times"]), allow_pickle=True)

        self.ori_train = np.load(os.path.join(
            middle_path, self.params["output_params"]["original_train"]), allow_pickle=True)
        self.ori_valid = np.load(os.path.join(
            middle_path, self.params["output_params"]["original_valid"]), allow_pickle=True)

        self.train_predict_g = np.load(os.path.join(
            middle_path, self.params["output_params"]["train_predict_G"]), allow_pickle=True)
        self.train_predict_g_d = np.load(os.path.join(
            middle_path, self.params["output_params"]["train_predict_G_D"]), allow_pickle=True)

        self.valid_predict_g = np.load(os.path.join(
            middle_path, self.params["output_params"]["valid_predict_G"]), allow_pickle=True)
        self.valid_predict_g_d = np.load(os.path.join(
            middle_path, self.params["output_params"]["valid_predict_G_D"]), allow_pickle=True)

        with open(os.path.join(self.params["output_params"]["result_dir"],
                               self.machine_id, 'data_info.json'), 'r') as f:
            self.data_info = json.load(f)

        return False

    def init_model(self, is_training=False):
        """Initializes the model"""
        model_path = model_path = os.path.join(
            self.params["output_params"]["model_dir"], self.machine_id)
        self.model = OnlineAd(x_dims=self.train_processed.shape[1],
                              max_epochs=self.params["usad_params"]["max_epochs"],
                              batch_size=self.params["usad_params"]["batch_size"],
                              z_dims=self.params["usad_params"]["z_dims"],
                              window_size=self.params["usad_params"]["window_size"])
        if not is_training:
            self.model.restore(os.path.join(model_path, 'shared_encoder'), os.path.join(model_path, 'decoder_G'),
                               os.path.join(model_path, 'decoder_D'))

    def predict(self, df, machine_id, detect_metrics, sli_metrics):
        """Runs online model predicting"""
        is_training = self.restore_model()
        if is_training:
            return [], [], None

        test_data_loader = DataLoader(self.config, is_training=is_training, machine_id=machine_id,
                                      test_data=df, detect_metrics=detect_metrics, sli_metrics=sli_metrics)
        self.test_processed = test_data_loader.return_test_data()
        self.all_features = test_data_loader.features
        self.scale = test_data_loader.scale
        self.ori_test = test_data_loader.ori_test

        self.init_model(is_training=is_training)

        self.test_predict_g, self.test_predict_g_d = self.model.predict(
            self.test_processed)

        self.x_all_processed = np.concatenate((self.train_processed[-len(self.train_predict_g_d):],
                                              self.valid_processed[-len(
                                                  self.valid_predict_g_d):],
                                              self.test_processed[-len(self.test_predict_g_d):]), axis=0)

        self.all_g = np.concatenate(
            (self.train_predict_g, self.valid_predict_g, self.test_predict_g), axis=0)
        self.all_g_d = np.concatenate(
            (self.train_predict_g_d, self.valid_predict_g_d, self.test_predict_g_d), axis=0)
        self.all_times = np.concatenate((self.train_times[-len(self.train_predict_g_d):],
                                         self.valid_times[-len(self.valid_predict_g_d):],
                                         test_data_loader.test_times[-len(self.test_predict_g_d):]), axis=0)

        np.save(os.path.join(self.params["output_params"]["result_dir"],
                self.params["output_params"]["all_processed_data"]), self.x_all_processed)
        np.save(os.path.join(self.params["output_params"]["result_dir"],
                self.params["output_params"]["all_G_D"]), self.all_g_d)
        np.save(os.path.join(self.params["output_params"]["result_dir"],
                self.params["output_params"]["all_times"]), self.all_times)
        np.save(os.path.join(self.params["output_params"]["result_dir"],
                self.params["output_params"]["features"]), self.all_features)

        self.post = PostProcessor(processed_data=self.x_all_processed,
                                  reconstruct_data=self.all_g_d,
                                  generate_data=self.all_g,
                                  data_info=self.data_info,
                                  all_features=self.all_features.tolist(),
                                  all_times=self.all_times,
                                  scaler=self.scale,
                                  ori_train=self.ori_train[-len(self.train_predict_g_d):],
                                  ori_valid=self.ori_valid[-len(self.valid_predict_g_d):],
                                  ori_test=self.ori_test[-len(self.test_predict_g_d):],
                                  sli_metrics=sli_metrics
                                  )
        abnormal_sli_metric = self.post.compute_score(score_type=self.params["usad_params"]["score_type"],
                                                      alpha=self.params["usad_params"]["alpha"])
        self.post.spot_fit(
            q=self.params["usad_params"]["q"], level=self.params["usad_params"]["level"])
        self.post.health_detection()

        y_pred, sli_pred = self.post.get_anomalies()
        return y_pred, sli_pred, abnormal_sli_metric

    def trigger_locate(self, machine_id, pearson_record):
        """Gets the possible root cause metrics of the anomaly"""
        now_time = datetime.now(timezone.utc).astimezone().astimezone()
        cause_list, anomaly_scores_list = self.post.locate_cause(
            machine_id, pearson_record)
        return cause_list, now_time, anomaly_scores_list

    def training(self, df, machine_id, detect_metrics, sli_metrics):
        """Runs online model training"""
        middle_path = os.path.join(
            self.params["output_params"]["middle_dir"], self.machine_id)
        model_path = os.path.join(
            self.params["output_params"]["model_dir"], self.machine_id)
        train_data_loader = DataLoader(self.config, is_training=True, machine_id=machine_id,
                                       train_data=df, detect_metrics=detect_metrics, sli_metrics=sli_metrics)
        df.to_csv(os.path.join(
            self.params["input_params"]["training_data_dir"], self.machine_id, "metric.csv"), index=False)
        self.train_processed, self.valid_processed = train_data_loader.return_data()
        np.save(os.path.join(
            middle_path, self.params["output_params"]["train_processed"]), self.train_processed)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["valid_processed"]), self.valid_processed)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["train_times"]), train_data_loader.train_times)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["valid_times"]), train_data_loader.valid_times)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["original_train"]), train_data_loader.ori_train)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["original_valid"]), train_data_loader.ori_valid)
        self.init_model(is_training=True)

        model_metrics = self.model.fit(
            self.train_processed, self.valid_processed)
        self.model.save(os.path.join(model_path, 'shared_encoder'), os.path.join(model_path, 'decoder_G'),
                        os.path.join(model_path, 'decoder_D'))

        self.train_predict_g, self.train_predict_g_d = self.model.predict(
            self.train_processed)
        self.valid_predict_g, self.valid_predict_g_d = self.model.predict(
            self.valid_processed)

        np.save(os.path.join(
            middle_path, self.params["output_params"]["train_predict_G"]), self.train_predict_g)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["train_predict_G_D"]), self.train_predict_g_d)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["valid_predict_G"]), self.valid_predict_g)
        np.save(os.path.join(
            middle_path, self.params["output_params"]["valid_predict_G_D"]), self.valid_predict_g_d)

        self.data_info = {
            'train_length': self.train_predict_g_d.shape[0],
            'valid_length': self.valid_predict_g_d.shape[0],
            'dataset': machine_id
        }
        data_json_file = os.path.join(
            self.params["output_params"]["result_dir"], self.machine_id, 'data_info.json')
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(data_json_file, flags, modes), 'w') as fout:
            json.dump(self.data_info, fout)

    def is_abnormal(self, y_pred):
        """Checks if existing abnormal or not"""
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        if len(y_pred) > POINTS_MINUTE:
            y_pred = y_pred[-POINTS_MINUTE:]

        if len(y_pred) < POINTS_MINUTE:
            logger.warning(
                f'The length of y_pred is less than {POINTS_MINUTE}')
            return False

        abnormal = sum([1 for y in y_pred if y > 0]) >= len(y_pred) * self.th

        if abnormal:
            logger.info(
                f'Detects abnormal events by {self.__class__.__name__}!')

        return abnormal
