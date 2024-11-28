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

import copy
from datetime import datetime, timedelta
from functools import reduce
from typing import List, Tuple, Union

import pandas as pd
from pandas import DataFrame

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI, ModelConfig, Feature
from anteater.core.ts import TimeSeries
from anteater.model.detector.base import Detector
from anteater.model.online_usad_model import OnlineUsadModel
from anteater.model.algorithms.pearson import select_relevant_kpi
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger
from anteater.utils.constants import POINTS_HOUR

def smooth_data(df, window=18):
    """Smooths metrics"""
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = df[col].rolling(window=window).mean().bfill().values
    return df


def check_nan(df):
    num_col = len(df.columns)
    count = 0
    for col in df.columns:
        if df[col].eq(0).all():
            count += 1
    if count / num_col > 0.95:
        logger.info(f"zero col count: {count}, {num_col}")
        return False
    return True

class UsadDetector(Detector):
    """The anomaly detector base class"""

    def __init__(self, data_loader: MetricLoader, config: ModelConfig, **kwargs):
        """The detector base class initializer"""
        super().__init__(data_loader, **kwargs)
        self.online_model = OnlineUsadModel(config)
        self.config = config
        self.anomaly_scores = None
        self.cause_list = None
        self.candidates = {}
        self.machine_training = {}
        self.detect_type = config.params.get("detect_type", "machine")

    def detect_machine_kpis(self, kpis: List[KPI], features: List[Feature], machine_id, key):
        """Executes anomaly detection on kpis"""
        detect_kpis = copy.deepcopy(features)
        sli_metrics = [kpi.metric for kpi in kpis]
        detect_kpis.extend(kpis)
        is_anomaly = False

        if key == "machine_id":
            detect_metrics = [kpi.metric for kpi in detect_kpis if "container" not in kpi.metric]
        else:
            # kpis only for pod
            detect_metrics = [kpi.metric for kpi in detect_kpis]

        detect_metrics.sort()

        min_minutes = self.online_model.get_min_predict_minutes()
        start, end = dt.last(minutes=min_minutes)

        if self.online_model.need_training(machine_id):
            hours = self.online_model.get_min_training_hours()
            train_df = self.get_training_data(
                start - timedelta(hours=hours), end, detect_kpis, machine_id, key)

            if train_df is None or len(train_df.index) == 0:
                logger.info('Got empty training data on machine: %s', machine_id)
                return [], is_anomaly
            elif len(train_df.index) < hours * POINTS_HOUR:
                logger.info('Less training data on machine: %s', machine_id)
                return [], is_anomaly
            else:
                logger.info('The shape of training data: %s on machine: %s',
                            str(train_df.shape), machine_id)
                self.online_model.train(train_df, machine_id, detect_metrics, sli_metrics)

        anomalies = []
        for machine_id, x_df in self.get_inference_data(start, end, detect_kpis, machine_id, key):
            if not check_nan(x_df):
               logger.info(f"{machine_id}, Metric empty, exit detection ********************************")
               return anomalies, is_anomaly

            _, last_5_min_x_dfs = self.get_inference_data(start - timedelta(minutes=5), end, detect_kpis, machine_id, key)[0]
            y_pred, sli_pred, abnormal_sli_metric = self.online_model.predict(last_5_min_x_dfs, x_df, machine_id,
                                                                              detect_metrics, sli_metrics)

            if len(y_pred) == 0:
                logger.info(f'***** len(y_pred) == 0')
                return anomalies, is_anomaly

            if machine_id not in self.machine_training:
                self.machine_training[machine_id] = True

            pearson_record = select_relevant_kpi(x_df, x_df[abnormal_sli_metric])
            cause_list, now_time, anomaly_scores_list = self.online_model.trigger_locate(pearson_record)

            if self.online_model.is_abnormal(sli_pred, machine_id):
                is_anomaly = True

            if len(cause_list) == 0 or len(anomaly_scores_list) == 0:
                if len(cause_list) == 0:
                    logger.info('cause_list is 0')
                if len(anomaly_scores_list) == 0:
                    logger.info('anomaly_scores_list is 0')
                return anomalies, is_anomaly

            self.anomaly_scores = copy.deepcopy(anomaly_scores_list[0])
            self.cause_list = copy.deepcopy(cause_list[0])
            anomalies.extend(self.find_abnormal_kpis(machine_id, key, kpis, features, anomaly_scores_list[0], top_n=1))

            new_features = copy.deepcopy(features)
            candidates = [c["metric"] for c in cause_list[0]["Resource"]["cause_metrics"]]
            remove_features = [f for f in new_features if f.metric not in candidates]
            for r in remove_features:
                new_features.remove(r)

            anomalies = self.get_usad_root_causes(anomalies, new_features, self.anomaly_scores, key)
            anomalies = self.cal_usad_kpi_anomaly_score(anomalies, self.anomaly_scores)

            for anomaly in anomalies:
                anomaly.is_anomaly = is_anomaly

        return anomalies, is_anomaly

    def _get_description(self, metric):
        description = self.data_loader.metricinfo.get_zh(metric)

        return description

    def find_abnormal_kpis(self, machine_id: str, key: str, kpis: List[KPI], features: List[Feature], scores, top_n=1):
        """Find abnormal kpis when detected anomaly events"""
        sli_metrics = [kpi.metric for kpi in kpis]
        detect_kpis = copy.deepcopy(kpis)
        detect_metrics = {kpi.metric: kpi for kpi in detect_kpis}

        ts_scores = []
        for kpi in kpis:
            description = self._get_description(kpi.metric)
            tmp_ts_scores, metric_similar = self.cal_usad_anomaly_score(kpi.metric, description, machine_id, float(scores[kpi.metric]), top_n=1, key=key)
            if tmp_ts_scores:
                for _ts_score in tmp_ts_scores:
                    _ts_score.score = _ts_score.score  # if check_trend(_ts_score.ts.values, kpi.atrend) else 0
                ts_scores.extend(tmp_ts_scores)

        # 检测此处的score异常, 未取等号
        ts_scores = [v for v in ts_scores if v.score >= 0]
        ts_scores = sorted(ts_scores, key=lambda x: x.score, reverse=True)
        ts_scores = ts_scores[: top_n]

        description = self._get_description(_ts_score.ts.metric)
        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts_score.ts.metric,
                labels=_ts_score.ts.labels,
                score=_ts_score.score,
                entity_name=detect_metrics[_ts_score.ts.metric.split(
                    "@")[0]].entity_name,
                description=description
            )
            for _ts_score in ts_scores
        ]

        return anomalies

    # todo 待改造
    def cal_metric_ab_score(self, metric: str, machine_id: str) \
            -> List[Tuple[TimeSeries, int]]:
        """Calculates metric abnormal scores based on sr model"""
        start, end = dt.last(minutes=10)
        ts_list = self.data_loader.get_metric(
            start, end, metric, machine_id=machine_id, )
        point_count = self.data_loader.expected_point_length(start, end)
        ts_scores = []
        for _ts in ts_list:
            if sum(_ts.values) == 0 or \
               len(_ts.values) < point_count * 0.9 or\
               len(_ts.values) > point_count * 1.5 or \
               all(x == _ts.values[0] for x in _ts.values):
                score = 0
            else:
                score = max(_ts.values)

            ts_scores.append((_ts, score))

        return ts_scores

    def get_training_data(self, start: datetime, end: datetime, metrics: List[str], machine_id: str, keys=None)\
            -> Union[DataFrame, None]:
        """Get the training data to support model training"""
        logger.info('Get training data from %s to %s on %s!',
                    start.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    end.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    machine_id)
        _, x_dfs = self.get_dataframe(start, end, metrics, machine_ids=[machine_id], keys=[keys])

        if not x_dfs:
            return None
        else:
            return x_dfs[0]
    def get_inference_data(self, start: datetime, end: datetime, kpis: List[KPI], machine_id=None, keys=None)\
            -> List[Tuple[str, DataFrame]]:
        """Get data for the model inference and prediction

        """
        logger.info('Get inference data during %s to %s on %s!',
                    start.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    end.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    machine_id)
        if machine_id is None:
            ids, x_dfs = self.get_dataframe(start, end, kpis)
        else:
            ids, x_dfs = self.get_dataframe(
                start, end, kpis, machine_ids=[machine_id], keys=[keys])

        return list(zip(ids, x_dfs))

    def get_dataframe(self, start, end, kpis, machine_ids=None, keys=None):
        """Gets the features during a period seperated by machine ids"""
        metrics = [kpi.metric for kpi in kpis]

        if machine_ids is None:
            machine_ids = self.get_unique_machine_id(start, end, kpis)
            pod_ids = self.get_unique_pod_id(start, end, kpis)

            keys = len(machine_ids) * ["machine_id"]
            keys += len(pod_ids) * ["pod_id"]
            entity_ids = machine_ids + pod_ids
            if not machine_ids:
                logger.warning(
                    "Cannot get unique machine ids from Prometheus!")
        else:
            entity_ids = machine_ids

        dataframes = []
        for entity_id, key in zip(entity_ids, keys):
            metric_dfs = []
            metrics_list = []

            for metric in metrics:
                if key == 'machine_id':
                    if 'container' in metric:
                        continue
                    ts_list = self.data_loader. \
                        get_metric(start, end, metric, operator='avg',
                                   keys="machine_id", machine_id=entity_id)
                else:
                    ts_list = self.data_loader. \
                        get_metric(start, end, metric, operator='avg',
                                    keys="pod_id", pod_id=entity_id)

                if len(ts_list) > 1:
                    raise ValueError(f'Got multiple time_series based'
                                     f'on machine id: {len(ts_list)}')

                if ts_list:
                    ts_list = ts_list[0]
                else:
                    ts_list = TimeSeries(metric, {}, [], [])

                if ts_list.to_df().columns[0] not in metrics_list:
                    metrics_list.append(ts_list.to_df().columns[0])
                    metric_dfs.append(ts_list.to_df())

            df = reduce(lambda left, right: pd.DataFrame(left).join(right, how='outer'), metric_dfs)
            df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
            df["timestamp"] = df.index.to_list()
            dataframes.append(df)

        return machine_ids, dataframes

    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) -> List[Anomaly]:
        """Multiple processes detect anomalies on individual machines"""
        logger.info(f'Execute model: {self.__class__.__name__}!')
        min_minutes = self.online_model.get_min_predict_minutes()
        start, end = dt.last(minutes=min_minutes)
        metrics = [k.metric for k in kpis]

        entity_ids = []
        entity_keys = []
        if self.detect_type == "machine":
            # machine
            machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
            entity_keys.extend(['machine_id'] * len(machine_ids))
            entity_ids.extend(machine_ids)
        else:
            # pod
            pod_ids = self.data_loader.get_unique_pods(start, end, metrics)
            entity_keys.extend(['pod_id'] * len(pod_ids))
            entity_ids.extend(pod_ids)

        if not entity_ids:
            logger.warning('Empty entity_ids, RETURN!')

        anomalies = []
        results = []

        logger.info(f"Detected entity number {len(entity_ids)}: {entity_ids}")
        for _id, key in zip(entity_ids, entity_keys):
            results.append(self.detect_machine_kpis(kpis, features, _id, key))
            logger.info(f"Finish test machine {_id} ********************")

        is_any_anomaly = False
        for r in results:
            if r[1]:
                is_any_anomaly = True
                break

        # 任一pod异常上传异常事件
        if is_any_anomaly:
            for r in results:
                anomalies.extend(r[0])

        return anomalies
