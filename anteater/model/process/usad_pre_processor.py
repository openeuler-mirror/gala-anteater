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

import warnings
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')


class PreProcessor:
    @staticmethod
    def clip_transform(value, alpha, mean=None, std=None):
        if mean is None:
            mean = np.mean(value, axis=0)
        if std is None:
            std = np.std(value.astype(np.float64), axis=0).astype(np.float32)
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
