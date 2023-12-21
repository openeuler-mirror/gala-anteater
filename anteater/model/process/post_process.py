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
import math
import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict

from anteater.model.algorithms.spot import Spot
from anteater.core.kpi import ModelConfig
from anteater.model.process.usad_health import HealthMulti
from anteater.utils.log import logger

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

class PostProcessor:
    def __init__(self, config: ModelConfig, machine_id, processed_data, reconstruct_data, generate_data, data_info,
                 all_features, all_times, scaler, ori_train, ori_valid, ori_test, sli_metrics):
        self.machine_id = machine_id
        self.params = config.params
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

    def find_abnormal_sli_metric(self, sli_idx: List) -> str:
        metric_reconstruct_error = np.sum(
            self.reconstruct_error[self.data_info['train_length'] + self.data_info['valid_length']:], axis=0)
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

    def compute_score(self, df, machine_id, metrics_to_delete):
        """Calculates the anomaly score by reconstruction error"""

        score_type = self.params["usad"]["score_type"]
        alpha = self.params["usad"]["alpha"]

        n_port = len(self.all_features)
        scale = np.array([1] * n_port)
        sli_scale = np.array([1] * n_port)
        sli_idx = []

        for sli_metric in self.sli_metrics:
            if sli_metric not in metrics_to_delete and sli_metric in df.columns.values.tolist():
                sli_metric_tmp_arr = np.array(df[sli_metric])
                if np.count_nonzero(sli_metric_tmp_arr) > 0:
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

        self.sli_reconstruct_error = self.reconstruct_error[:, sli_idx] # [all, 1]
        abnormal_sli_metric = self.find_abnormal_sli_metric(sli_idx) # 96/111

        logger.info(f"{machine_id}: compute_score sli idx: {sli_idx}/{self.true_processed_data.shape[-1]}, {abnormal_sli_metric}")

        sli_scores = np.sum(self.sli_reconstruct_error, axis=1)
        sli_diff_scores = np.diff(sli_scores)
        sli_diff_scores_0 = sli_diff_scores[0]
        scores = np.sum(self.reconstruct_error, axis=1)
        diff_scores = np.diff(scores)
        diff_scores_0 = diff_scores[0]
        self.scores = np.insert(diff_scores, 0, diff_scores_0)
        self.sli_scores = np.insert(sli_diff_scores, 0, sli_diff_scores_0)

        return abnormal_sli_metric

    def spot_fit(self, threshold=None):
        """Obtains dynamic threshold for anomaly scores by the SPOT algorithm"""
        # spot params
        q = self.params["usad"]["q"]
        level = self.params["usad"]["level"]

        train_data = np.copy(self.scores[:self.data_info['train_length']])
        fit_data = self.scores[self.data_info['train_length'] + self.data_info['valid_length']:]

        s = Spot(q)
        s.initialize(train_data, level=level)
        self.spot_detect_res = s.run(fit_data)

        self.spot_detect_res["upper_thresholds"] = self.spot_detect_res["thresholds"]
        self.spot_detect_res["lower_thresholds"] = self.spot_detect_res["thresholds"]

        # sli metrics
        sli_train_data = np.copy(self.sli_scores[:self.data_info['train_length']])
        sli_fit_data = self.sli_scores[self.data_info['train_length'] + self.data_info['valid_length']:]

        sli_s = Spot(q)
        sli_s.initialize(sli_train_data, level=level)
        self.sli_spot_detect_res = sli_s.run(sli_fit_data)

        self.sli_spot_detect_res["upper_thresholds"] = self.sli_spot_detect_res["thresholds"]
        self.sli_spot_detect_res["lower_thresholds"] = self.sli_spot_detect_res["thresholds"]

        return self.spot_detect_res, self.sli_spot_detect_res

    def health_detection(self):
        """Filters reported anomalies through health detection"""
        score_std = np.std(self.scores[:self.data_info['train_length']])
        health = HealthMulti(score_std, None)
        for i in range(len(self.spot_detect_res['upper_thresholds'])):
            x = self.true_processed_data[(self.data_info['train_length'] + self.data_info['valid_length']):][i]
            err = self.scores[(self.data_info['train_length'] + self.data_info['valid_length']):][i]
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
            sli_x = self.sli_true_processed_data[(self.data_info['train_length'] + self.data_info['valid_length']):][sli_i]
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

    def locate_cause(self, metrics_to_delete, pearson_record, threshold=None):
        """Locates the possible root cause of the anomaly"""
        anomalies_all = np.array(self.spot_detect_res['alarms']) + self.data_info['train_length'] + self.data_info[
            'valid_length']

        if self.sli_spot_detect_res['alarms']:
            anomalies_sli = np.array(self.sli_spot_detect_res['alarms']) + self.data_info['train_length'] + \
                            self.data_info['valid_length']
        else:
            anomalies_sli = np.arange(24) + self.data_info['train_length'] + self.data_info[
                'valid_length']
        anomalies = anomalies_sli
        # 这个地方只检测性能指标，可以改成机器指标和性能指标都检测
        # anomalies = np.unique(np.concatenate((anomalies_all, anomalies_sli))).astype(int)
        segment_list_l = get_segment_list(anomalies)

        anomalies_cause_dict = {}
        anomalies_cause_full_dict = {}
        np_error_rate_all = []

        for segment_l in segment_list_l:
            if len(segment_l) != 0:
                # seg_l = [i for i in segment_l if i in anomalies_sli or i in anomalies_all]
                seg_l = [i for i in segment_l if i in anomalies_sli]
                seg_cause_idx, np_err_rate_seg, seg_cause_idx_full = self.get_cause_weight(seg_l, threshold)
                np_error_rate_all.append(np_err_rate_seg)
                dict_key = " ".join([str(i) for i in segment_l])
                anomalies_cause_full_dict[dict_key] = {"cause_index": seg_cause_idx_full, "error_rate": np_err_rate_seg}
            else:
                logger.info('***********ERROR***********')
                seg_cause_idx = []
            anomalies_cause_dict[" ".join([str(i) for i in segment_l])] = seg_cause_idx

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
                if k_ in metrics_to_delete:
                    cur_cause_list[k_] = 0.0
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

    def get_cause_weight(self, segment_l, threshold=None, machine_id="", is_pm=False):
        """Calculation of root cause index weights for root cause localization"""
        idx_ = min(segment_l[0] + 10, len(self.reconstruct_error))
        segment_require = [i for i in range(segment_l[0], idx_)]
        if is_pm:
            middle_path = os.path.join(
                self.params["output_params"]["middle_dir"], machine_id)
            reconstruct_error = np.load(os.path.join(
                middle_path, self.params["output_params"]["reconstruct_error"]), allow_pickle=True)
            train_reconstruct_avg = np.mean(reconstruct_error[:self.data_info['train_length']], axis=0)
            train_reconstruct_std = np.std(reconstruct_error[:self.data_info['train_length']], axis=0)
            kpi_normalised = abs(reconstruct_error[segment_require, :] - train_reconstruct_avg)
        else:
            train_reconstruct_avg = np.mean(
                self.reconstruct_error[:self.data_info['train_length']], axis=0)
            train_reconstruct_std = np.std(
                self.reconstruct_error[:self.data_info['train_length']], axis=0)
            kpi_normalised = abs(
                self.reconstruct_error[segment_require, :] - train_reconstruct_avg)
        train_reconstruct_std[np.where(train_reconstruct_std == 0)[0]] = 1e-8

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

        anomalies = np.array(self.spot_detect_res['alarms']) + self.data_info['train_length'] + self.data_info[
            'valid_length']

        pred_anomaly_times = []
        for i in anomalies:
            pred_anomaly_times.append(self.all_times[i])

        test_times = self.all_times[(self.data_info['train_length'] + self.data_info['valid_length']):]

        result_df = pd.DataFrame({"timestamp": test_times, "ground_truth": np.zeros(test_times.size),
                                  "pred": np.zeros(test_times.size)})
        pred_index = result_df[result_df["timestamp"].isin(pred_anomaly_times)].index
        result_df.loc[pred_index, "pred"] = 1

        sli_anomalies = np.array(self.sli_spot_detect_res['alarms']) + self.data_info['train_length'] + self.data_info[
            'valid_length']
        # logger.info(f"get sli_anomalies: {sli_anomalies}")
        sli_pred_anomaly_times = []
        for sli_i in sli_anomalies:
            sli_pred_anomaly_times.append(self.all_times[sli_i])

        sli_test_times = self.all_times[(self.data_info['train_length'] + self.data_info['valid_length']):]

        sli_result_df = pd.DataFrame({"timestamp": sli_test_times, "ground_truth": np.zeros(sli_test_times.size),
                                      "pred": np.zeros(sli_test_times.size)})

        sli_pred_index = sli_result_df[sli_result_df["timestamp"].isin(
            sli_pred_anomaly_times)].index
        sli_result_df.loc[sli_pred_index, "pred"] = 1

        return result_df["pred"], sli_result_df["pred"]

    def get_scores(self):
        return self.scores, self.reconstruct_error

    def get_data(self):
        return self.true_processed_data, self.true_reconstruct_data