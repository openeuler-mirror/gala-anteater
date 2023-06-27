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
import pickle
import stat
import math
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from anteater.model.algorithms.spot import Spot
from anteater.utils.log import logger


def smooth_data(df, window=3):
    """Smooths metrics"""
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = df[col].rolling(window=window).mean().bfill().values
    return df


def sigmoid(z, alpha, offset=0.5):
    """Sigmoid activation functions"""
    y = 1 / (1 + np.exp(-alpha * (z - offset)))
    return y


def relu(x, alpha, offset, liner_criteria=None):
    """ReLU activation functions"""
    if liner_criteria is None:
        liner_criteria = 0.5
    if x > liner_criteria:
        return sigmoid(x, alpha=alpha, offset=offset)
    elif x <= 0:
        return 0
    else:
        return x


def get_segment_list(anomalies):
    if len(anomalies) == 0:
        return []
    result = []
    s = []
    for i in anomalies:
        if len(s) == 0 or s[-1] + 1 == i:
            s.append(i)
        else:
            result.append(s)
            s = []
            s.append(i)
    result.append(s)
    return result


class DataLoader:
    def __init__(self, config, is_training=False, **kwargs):
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

        self.sli_metrics = kwargs.get("sli_metrics")
        self.machine_id = kwargs.get("machine_id")
        self.detect_metrics = kwargs.get("detect_metrics")

        if is_training:
            self.raw_data = kwargs.get("train_data")
            self.__read_data(self.raw_data)
            self.save_scale()
        else:
            self.load_scale()
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

    def save_scale(self):
        """Saves the data preprocessing parameters"""
        pkl_file = os.path.join(
            self.config.params["output_params"]["result_dir"], self.machine_id, 'scaler.pkl')
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(pkl_file, flags, modes), 'wb') as fout:
            tmp = pickle.dumps(self.__scale)
            fout.write(tmp)

    def load_scale(self):
        """Loads the data preprocessing parameters"""
        pkl_file = os.path.join(
            self.config.params["output_params"]["result_dir"], self.machine_id, 'scaler.pkl')
        try:
            input_hal = open(pkl_file, 'rb')
        except EOFError:
            logger.error("EOFError: Load scale failed")
        else:
            self.__scale = pickle.load(input_hal)
            input_hal.close()

    def __read_data(self, raw_data):
        """Reads training data"""
        metric_raw = self.raw_data
        metric_candidates = [col for col in metric_raw.columns]

        metric = smooth_data(
            metric_raw, window=self.config.params["usad_params"]["smooth_window"])
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
            self.config.params["usad_params"][
                "preprocess_type"],
            clip_alpha=self.config.params["usad_params"]["clip_alpha"])

    def __read_test_data(self, df):
        """Reads test data"""
        metric_raw = df
        metric = smooth_data(
            metric_raw, window=self.config.params["usad_params"]["smooth_window"])

        channels_pd = metric[self.detect_metrics]
        time_stamp_str = metric["timestamp"]
        channels_pd['timestamp'] = pd.to_datetime(time_stamp_str).values

        test_df = channels_pd
        self.__test_times = test_df.timestamp.astype(str).tolist()
        self.__features = np.array(channels_pd.columns)[:-1]
        self.test_np = test_df.values[:, :-1]
        self.test_np, self.ori_test = PreProcessor.test_preprocess(
            self.__scale,
            self.test_np,
            self.config.params["usad_params"]["preprocess_type"],
            clip_alpha=self.config.params["usad_params"]["clip_alpha"])

    def __split_data(self, rawdata_df):
        """"Divides the training and validation data"""
        train_start = self.config.params["input_params"]["train_start"]
        train_end = self.config.params["input_params"]["train_end"]
        valid_end = self.config.params["input_params"]["valid_end"]
        if train_end is not None:
            train_df = rawdata_df.loc[rawdata_df.timestamp <
                                      pd.to_datetime(train_end)]
        else:
            train_df = rawdata_df
        if train_start is not None:
            train_df = train_df.loc[rawdata_df.timestamp >=
                                    pd.to_datetime(train_start)]
        if valid_end is not None:
            valid_df = rawdata_df.loc[rawdata_df.timestamp >= pd.to_datetime(
                train_end)]
            valid_df = valid_df.loc[rawdata_df.timestamp <
                                    pd.to_datetime(valid_end)]
        else:
            n = int(len(train_df) * 0.2)
            train_df, valid_df = train_df[:-n], train_df[-n:]
        return train_df, valid_df


class PostProcessor:
    def __init__(self, processed_data, reconstruct_data, generate_data, data_info,
                 all_features, all_times, scaler, ori_train, ori_valid, ori_test, sli_metrics):
        self.model_processed_data = processed_data
        self.model_reconstruct_data = reconstruct_data
        self.model_generate_data = generate_data
        self.data_info = data_info
        self.all_features = all_features
        self.all_times = all_times
        self.scaler = scaler
        self.sli_metrics = sli_metrics
        self.cause_info = []
        self.cause_index = {}
        self.cause_time_seg_map = defaultdict(list)
        self.scores = None
        self.sli_scores = None
        self.reconstruct_error = None
        self.sli_reconstruct_error = None
        self.spot_detect_res = None
        self.sli_spot_detect_res = None
        self.health_detect_res = {}
        self.sli_health_detect_res = {}
        self.upper_bounding = None
        self.sli_upper_bounding = None

        self.true_processed_data = np.concatenate(
            (ori_train, ori_valid, ori_test), axis=0)
        self.true_reconstruct_data = None
        self.sli_true_processed_data = None
        self.sli_true_reconstruct_data = None

    def find_abnormal_sli_metric(self, sli_idx):
        metric_reconstruct_error = np.sum(self.reconstruct_error, axis=0)
        max_reconstruct_error = None
        max_idx = -1
        for idx in sli_idx:
            if max_reconstruct_error is None:
                max_reconstruct_error = metric_reconstruct_error[idx]
                max_idx = idx
            else:
                if metric_reconstruct_error[idx] > max_reconstruct_error:
                    max_reconstruct_error = metric_reconstruct_error[idx]
                    max_idx = idx
        return self.all_features[max_idx]

    def compute_score(self, score_type, alpha):
        """Calculates the anomaly score by reconstruction error"""
        n_port = len(self.all_features)
        scale = np.array([1] * n_port)
        sli_scale = np.array([1] * n_port)
        sli_idx = []
        for sli_metric in self.sli_metrics:
            sli_idx.append(self.all_features.index(sli_metric))

        self.sli_true_processed_data = self.true_processed_data[:, sli_idx]

        if score_type == 'common':
            self.reconstruct_error = alpha * np.square(self.model_processed_data - self.model_generate_data) + \
                (1 - alpha) * np.square(self.model_processed_data - self.model_reconstruct_data)
        elif score_type == 'percentage':
            self.reconstruct_error = np.abs(
                self.model_reconstruct_data - self.model_processed_data) / scale
        elif score_type == 'power10':
            self.reconstruct_error = np.abs(
                np.power(10, self.model_processed_data / scale) - np.power(10, self.model_reconstruct_data / scale))
        else:
            raise NotImplementedError()

        self.sli_reconstruct_error = self.reconstruct_error[:, sli_idx]
        abnormal_sli_metric = self.find_abnormal_sli_metric(sli_idx)
        sli_scores = np.sum(self.sli_reconstruct_error, axis=1)
        sli_diff_scores = np.diff(sli_scores)
        sli_diff_scores_0 = sli_diff_scores[0]
        scores = np.sum(self.reconstruct_error, axis=1)
        diff_scores = np.diff(scores)
        diff_scores_0 = diff_scores[0]
        self.scores = np.insert(diff_scores, 0, diff_scores_0)
        self.sli_scores = np.insert(sli_diff_scores, 0, sli_diff_scores_0)
        return abnormal_sli_metric

    def spot_fit(self, q=1e-3, level=0.98, threshold=None):
        """Obtains dynamic threshold for anomaly scores by the SPOT algorithm"""
        train_data = np.copy(self.scores[:self.data_info['train_length']])

        s = Spot(q)
        s.fit(train_data,
              self.scores[self.data_info['train_length'] + self.data_info['valid_length']:])
        s.initialize(level=level, verbose=False)

        self.spot_detect_res = s.run()
        self.spot_detect_res["upper_thresholds"] = self.spot_detect_res["thresholds"]
        self.spot_detect_res["lower_thresholds"] = self.spot_detect_res["thresholds"]

        # sli metrics
        sli_train_data = np.copy(
            self.sli_scores[:self.data_info['train_length']])
        sli_s = Spot(q)
        sli_s.fit(sli_train_data,
                  self.sli_scores[self.data_info['train_length'] + self.data_info['valid_length']:])
        sli_s.initialize(level=level, verbose=False)
        self.sli_spot_detect_res = sli_s.run()
        self.sli_spot_detect_res["upper_thresholds"] = self.sli_spot_detect_res["thresholds"]
        self.sli_spot_detect_res["lower_thresholds"] = self.sli_spot_detect_res["thresholds"]
        return self.spot_detect_res, self.sli_spot_detect_res

    def health_detection(self):
        """Filters reported anomalies through health detection"""
        score_std = np.std(self.scores[:self.data_info['train_length']])
        health = HealthMulti(score_std, None)
        for i in range(len(self.spot_detect_res['upper_thresholds'])):
            x = self.true_processed_data[(
                self.data_info['train_length'] + self.data_info['valid_length']):][i]
            err = self.scores[(self.data_info['train_length'] +
                               self.data_info['valid_length']):][i]
            thre = self.spot_detect_res['upper_thresholds'][i]
            health.fit(i, x, err, thre)

        health_score_info = np.array(health.health_score_info)
        health_anomalies = health.predict(threshold=0.2)
        self.health_detect_res['health_score'] = health_score_info[:, 0]
        self.health_detect_res['deviation_score'] = health_score_info[:, 1]
        self.health_detect_res['sustained_score'] = health_score_info[:, 2]
        self.health_detect_res['alarms'] = health_anomalies

        # sli metrics
        sli_score_std = np.std(self.sli_scores[:self.data_info['train_length']])
        sli_health = HealthMulti(sli_score_std, None)
        for sli_i in range(len(self.sli_spot_detect_res['upper_thresholds'])):
            sli_x = self.sli_true_processed_data[(
                self.data_info['train_length'] + self.data_info['valid_length']):][sli_i]
            sli_err = self.sli_scores[(
                self.data_info['train_length'] + self.data_info['valid_length']):][sli_i]
            sli_thre = self.sli_spot_detect_res['upper_thresholds'][sli_i]
            sli_health.fit(sli_i, sli_x, sli_err, sli_thre)

        sli_health_score_info = np.array(sli_health.health_score_info)
        sli_health_anomalies = sli_health.predict(threshold=0.2)
        self.sli_health_detect_res['health_score'] = sli_health_score_info[:, 0]
        self.sli_health_detect_res['deviation_score'] = sli_health_score_info[:, 1]
        self.sli_health_detect_res['sustained_score'] = sli_health_score_info[:, 2]
        self.sli_health_detect_res['alarms'] = sli_health_anomalies

    def locate_cause(self, machine_id, pearson_record, threshold=None):
        """Locates the possible root cause of the anomaly"""
        anomalies_l = self.health_detect_res['alarms']
        anomalies = np.array(anomalies_l) + self.data_info['train_length'] + self.data_info['valid_length']
        segment_list_l = get_segment_list(anomalies)
        anomalies_spot = np.array(self.spot_detect_res['alarms']) + self.data_info['train_length'] + self.data_info[
            'valid_length']
        anomalies_cause_dict = {}
        anomalies_cause_full_dict = {}
        np_error_rate_all = []
        for segment_l in segment_list_l:
            if len(segment_l) != 0:
                seg_l = [i for i in segment_l if i in anomalies_spot]
                seg_cause_idx, np_err_rate_seg, seg_cause_idx_full = self.get_cause_weight(seg_l, threshold)
                np_error_rate_all.append(np_err_rate_seg)
                dict_key = " ".join([str(i) for i in segment_l])
                anomalies_cause_full_dict[dict_key] = {
                    "cause_index": seg_cause_idx_full, "error_rate": np_err_rate_seg
                }
            else:
                seg_cause_idx = []
            anomalies_cause_dict[" ".join(
                [str(i) for i in segment_l])] = seg_cause_idx
        np_error_rate_all = np.array(np_error_rate_all).T
        if len(np_error_rate_all) != 0:
            df_error_rate_all = pd.DataFrame(np_error_rate_all, columns=[str(s_l[0]) for s_l in segment_list_l])
        for k, v_l in anomalies_cause_dict.items():
            for v in v_l:
                self.cause_time_seg_map[v].append(k)
        cause_list = []
        anomaly_scores_list = []
        for k, v in anomalies_cause_full_dict.items():
            k_list = k.split(" ")
            anomaly_start = self.all_times[int(k_list[0])]
            anomaly_end = self.all_times[int(k_list[-1])]
            features_name = [self.all_features[item] for item in v["cause_index"]]
            error_rate = [v["error_rate"][item] for item in v["cause_index"]]
            cur_cause_list = dict(zip(features_name, error_rate))
            for k_ in cur_cause_list:
                p_score = abs(pearson_record[k_]) if k_ in pearson_record else 0.0
                cur_cause_list[k_] *= p_score
            anomaly_scores_list.append(cur_cause_list)

            def construct_metric_info(metric_name, metric_score, metric_labels=None):
                metric_info = {"metric": metric_name, "score": metric_score}
                return metric_info

            cur_result_json = {}
            top1_metric = construct_metric_info(features_name[0], error_rate[0])
            cur_result_json["Resource"] = top1_metric
            cur_cause_list = [construct_metric_info(
                metric_name, metric_score) for metric_name, metric_score in cur_cause_list.items()]
            cur_result_json["Resource"]["cause_metrics"] = cur_cause_list
            cause_list.append(cur_result_json)

        return cause_list, anomaly_scores_list

    def get_cause_weight(self, segment_l, threshold=None):
        """Calculation of root cause index weights for root cause localization"""
        train_reconstruct_avg = np.mean(
            self.reconstruct_error[:self.data_info['train_length']], axis=0)
        train_reconstruct_std = np.std(
            self.reconstruct_error[:self.data_info['train_length']], axis=0)
        train_reconstruct_std[np.where(train_reconstruct_std == 0)[0]] = 1e-8

        segment_require = segment_l[:10]
        kpi_normalised = abs(
            self.reconstruct_error[segment_require, :] - train_reconstruct_avg)
        param_coe = 3
        array_coe = np.array(
            [math.log(1 / (i / 10 + pow(math.e, -param_coe))) / param_coe for i in range(10)])
        array_kpi_degree = np.multiply(
            kpi_normalised.T, array_coe[:kpi_normalised.shape[0]])
        array_kpi_degree = array_kpi_degree.sum(1)
        array_kpi_degree_rate = array_kpi_degree / \
            (array_kpi_degree.sum(0))
        if threshold:
            alarm_index = np.where(array_kpi_degree_rate > threshold)[0]
        else:
            alarm_index_full = np.argsort(array_kpi_degree)[::-1]
            alarm_index = alarm_index_full[:2]
        return alarm_index, array_kpi_degree_rate, alarm_index_full

    def get_anomalies(self):
        """Gets the results predicted by the model"""
        try:
            anomalies = np.array(self.health_detect_res['alarms']) + (
                self.data_info['train_length'] + self.data_info['valid_length'])
        except KeyError as e:
            if 'alarms' in self.health_detect_res:
                logger.error("%s: Get data info error", e)
            else:
                logger.error("%s: Get health detect result error", e)
        pred_anomaly_times = []
        for i in anomalies:
            pred_anomaly_times.append(self.all_times[i])

        test_times = self.all_times[(
            self.data_info['train_length'] + self.data_info['valid_length']):]

        result_df = pd.DataFrame({"timestamp": test_times, "ground_truth": np.zeros(test_times.size),
                                  "pred": np.zeros(test_times.size)}, dtype=int)
        pred_index = result_df[result_df["timestamp"].isin(
            pred_anomaly_times)].index
        result_df.loc[pred_index, "pred"] = 1

        # sli metrics
        try:
            sli_anomalies = np.array(self.sli_health_detect_res['alarms']) + (
                self.data_info['train_length'] + self.data_info['valid_length'])
        except KeyError as e:
            if 'alarms' in self.sli_health_detect_res:
                logger.error("%s: Get data info error", e)
            else:
                logger.error("%s: Get sli health detect result error", e)
        sli_pred_anomaly_times = []
        for sli_i in sli_anomalies:
            sli_pred_anomaly_times.append(self.all_times[sli_i])

        sli_test_times = self.all_times[(
            self.data_info['train_length'] + self.data_info['valid_length']):]

        sli_result_df = pd.DataFrame({"timestamp": sli_test_times, "ground_truth": np.zeros(sli_test_times.size),
                                      "pred": np.zeros(sli_test_times.size)}, dtype=int)

        sli_pred_index = sli_result_df[sli_result_df["timestamp"].isin(
            sli_pred_anomaly_times)].index
        sli_result_df.loc[sli_pred_index, "pred"] = 1

        return result_df["pred"], sli_result_df["pred"]

    def get_scores(self):
        return self.scores, self.reconstruct_error

    def get_data(self):
        return self.true_processed_data, self.true_reconstruct_data


class PreProcessor:
    @staticmethod
    def clip_transform(value, alpha, mean=None, std=None):
        if mean is None:
            mean = np.mean(value, axis=0)
        if std is None:
            std = np.std(value, axis=0)
            for x in std:
                if x < 1e-4:
                    x = 1
        for i in range(value.shape[0]):
            # compute clip value: (mean - a * std, mean + a * std)
            clip_value = mean + alpha * std
            temp = value[i] < clip_value
            value[i] = temp * value[i] + (1 - temp) * clip_value
            clip_value = mean - alpha * std
            temp = value[i] > clip_value
            value[i] = temp * value[i] + (1 - temp) * clip_value
            std = np.maximum(std, 1e-5)  # to avoid std -> 0
            value[i] = (value[i] - mean) / std  # normalization
        return value, mean, std

    @classmethod
    def preprocess_mix_max100(cls, np_train, np_valid, np_test, metric_length, bandwidth_l):
        np_train = np.asarray(np_train, dtype=np.float32)
        np_valid = np.asarray(np_valid, dtype=np.float32)
        np_test = np.asarray(np_test, dtype=np.float32)

        metric_train = np_train[:, :metric_length]
        metric_valid = np_valid[:, :metric_length]
        metric_test = np_test[:, :metric_length]

        log_train = np_train[:, metric_length:]
        log_valid = np_valid[:, metric_length:]
        log_test = np_test[:, metric_length:]

        # log minmax
        scale_log = MinMaxScaler()
        scale_log = scale_log.fit(log_train)
        log_train = scale_log.transform(log_train)
        log_valid = scale_log.transform(log_valid)
        log_test = scale_log.transform(log_test)

        # metric max100
        cpu_mem_cols_len = metric_length - len(bandwidth_l)
        scale_metric = np.array([100] * cpu_mem_cols_len + bandwidth_l)
        metric_train = metric_train / scale_metric
        metric_valid = metric_valid / scale_metric
        metric_test = metric_test / scale_metric

        np_train = np.concatenate([metric_train, log_train], axis=1)
        np_valid = np.concatenate([metric_valid, log_valid], axis=1)
        np_test = np.concatenate([metric_test, log_test], axis=1)
        return np_train, np_valid, np_test, scale_metric, scale_log

    @classmethod
    def preprocess(cls, df_train, df_valid, pro_type, clip_alpha):

        df_train = np.asarray(df_train, dtype=np.float32)
        df_valid = np.asarray(df_valid, dtype=np.float32)

        ori_train, ori_valid = df_train, df_valid

        if pro_type == "minmax":
            scale = MinMaxScaler()
            scale = scale.fit(df_train)
            df_train = scale.transform(df_train)
            df_valid = scale.transform(df_valid)

        elif pro_type == "minmax_all":
            df_all = np.concatenate((df_train, df_valid), axis=0)
            scale = MinMaxScaler()
            scale = scale.fit(df_all)
            df_train = scale.transform(df_train)
            df_valid = scale.transform(df_valid)

        elif pro_type == "standard":
            scale = StandardScaler().fit(df_train)
            df_train = scale.transform(df_train)
            df_valid = scale.transform(df_valid)

        elif pro_type == "standard_respective":
            scale = StandardScaler().fit(df_train)
            df_train = scale.transform(df_train)
            df_valid = scale.transform(df_valid)

        elif pro_type == "clip" and clip_alpha:
            alpha = clip_alpha
            df_train, _mean, _std = cls.clip_transform(df_train, alpha)
            scale = {"mean": _mean, "std": _std}
            valid_res, _mean, _std = cls.clip_transform(df_valid, alpha, _mean, _std)
            df_valid = valid_res

        else:
            raise ValueError('need choose preprocess method')
        return df_train, df_valid, scale, ori_train, ori_valid

    @classmethod
    def test_preprocess(cls, scale, df_test, pro_type, clip_alpha):
        df_test = np.asarray(df_test, dtype=np.float32)
        ori_test = df_test

        if pro_type == "minmax":
            df_test = scale.transform(df_test)
        elif pro_type == "minmax_all":
            df_test = scale.transform(df_test)
        elif pro_type == "standard":
            df_test = scale.transform(df_test)
        elif pro_type == "standard_respective":
            scale = StandardScaler().fit(df_test)
            df_test = scale.transform(df_test)
        elif pro_type == "clip" and clip_alpha:
            alpha = clip_alpha
            test_res, _mean, _std = cls.clip_transform(df_test, alpha, scale["mean"], scale["std"])
            df_test = test_res
        else:
            raise ValueError('need choose preprocess method')
        return df_test, ori_test


class HealthMulti:
    def __init__(self, train_std, criterion_np=None):
        """Multi-metrics health detection"""

        self.train_std = train_std
        self.criterion_np = criterion_np
        self.window_length_limit = 5
        self.anomaly_window = {}
        self.sustained_healthy_number = 0
        self.anomaly_happen = False

        self.dev_weight = 0.5
        self.dur_weight = 0.5

        self.update_flag = False
        self.alarms = []
        self.health_info = {}

    @property
    def health_score_info(self):
        res = []
        for v in self.health_info.values():
            res.append(v)
        return res

    def get_weight(self, x_np, error, threshold):
        """Gets dev_weight and dur_weight"""
        if self.criterion_np is not None:
            if len(np.where(x_np >= self.criterion_np)[0]) != 0:
                self.dev_weight = 1.0
                self.dur_weight = 0.0
            else:
                self.dev_weight = 0.0
                self.dur_weight = 1.0
        else:
            self.dev_weight = 0.0
            self.dur_weight = 1.0

    def get_deviation_score(self, x_np, error, threshold):
        """Gets deviation score"""
        if error >= threshold:
            ratio = 1.0
            if self.criterion_np is not None:
                if len(np.where(x_np >= self.criterion_np)[0]) != 0:
                    return 1.0
                else:
                    ratio = np.max(np.abs(x_np - threshold) /
                                   (self.criterion_np - threshold + 1e-4))
            ratio = min(ratio, 1.0)
            score = (error - threshold) / self.train_std
            score = relu(ratio, 14, 0.5) * sigmoid(score, 5 / 6)
        else:
            score = 0
        return score

    def get_sustained_score(self, index, error, threshold):
        """Gets sustained score"""
        if error > threshold:
            self.anomaly_window[index] = 1
            self.sustained_healthy_number = 0
            if len(self.anomaly_window) <= self.window_length_limit:
                if np.sum([v for k, v in self.anomaly_window.items()]) < 3:
                    score = 0
                else:
                    score = 1
                    self.anomaly_happen = True
            else:
                score = 1
                self.anomaly_happen = True

        else:
            self.sustained_healthy_number = self.sustained_healthy_number + 1
            if self.anomaly_happen:
                if self.sustained_healthy_number < 3:
                    score = 1
                    self.anomaly_window[index] = 0
                else:
                    score = 0
                    self.anomaly_window = {}
                    self.anomaly_happen = False
                    self.update_flag = False
            else:
                score = 0
                if self.sustained_healthy_number < 3:
                    self.anomaly_window[index] = 0
                else:
                    self.anomaly_window = {}
                    self.update_flag = False
                    self.anomaly_happen = False

        return score

    def fit(self, index, x_np, error, threshold):
        """Gets health score"""
        deviation_score = self.get_deviation_score(x_np, error, threshold)
        sustained_score = self.get_sustained_score(index, error, threshold)
        self.get_weight(x_np, error, threshold)
        health_score = 1 - (deviation_score * self.dev_weight +
                            sustained_score * self.dur_weight)
        self.health_info[index] = [
            health_score, deviation_score, sustained_score, self.dur_weight
        ]
        self.update(health_score, sustained_score)

    def predict(self, threshold):
        """Gets anomaly alarms"""
        for index, score_l in self.health_info.items():
            if score_l[0] < threshold:
                self.alarms.append(index)
        return self.alarms

    def update(self, health_score, new_sustained_score):
        """Adjusts health score"""
        if health_score < 0.2 and not self.update_flag:
            for _id in self.anomaly_window.keys():
                origin_health_score_info = self.health_info.get(_id, [])
                if len(origin_health_score_info) == 0:
                    self.update_flag = False
                    return
                new_health_score = 1 - \
                    (origin_health_score_info[1] * self.dev_weight +
                     new_sustained_score * self.dur_weight)
                origin_health_score_info[0] = new_health_score
                origin_health_score_info[2] = new_sustained_score
                self.health_info[_id] = origin_health_score_info
            self.update_flag = True
        else:
            pass