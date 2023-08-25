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
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


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

    def detect_machine_kpis(self, kpis: List[KPI], features: List[Feature], machine_id):
        """Executes anomaly detection on kpis"""
        metrics = [k.metric for k in kpis] + [f.metric for f in features]
        min_minutes = self.online_model.get_min_predict_minutes()
        start, end = dt.last(minutes=min_minutes)
        if self.online_model.need_training(machine_id):
            hours = self.online_model.get_min_training_hours()
            train_df = self.get_training_data(
                start - timedelta(hours=hours), end, metrics, machine_id)
            if train_df is None or len(train_df.index) == 0:
                logger.info('Got empty training data on machine: %s', machine_id)
                return []
            elif len(train_df.index) < hours * 60:
                logger.info('Less training data on machine: %s', machine_id)
                return []
            else:
                logger.info('The shape of training data: %s on machine: %s',
                            str(train_df.shape), machine_id)
                self.online_model.train(train_df, machine_id)

        anomalies = []
        for machine_id, x_df in self.get_inference_data(start, end, metrics, machine_id):
            y_pred = self.online_model.predict(x_df, machine_id)
            if self.online_model.is_abnormal(y_pred):
                anomalies.extend(self.find_abnormal_kpis(machine_id, kpis))

        if anomalies:
            logger.info('%d anomalies was detected on %s.',
                        len(anomalies), self.__class__.__name__)
            anomalies = self.find_root_causes(anomalies, features, top_n=3)

        return anomalies

    def find_abnormal_kpis(self, machine_id: str, kpis: List[KPI]):
        """Find abnormal kpis when detected anomaly events"""
        detect_kpis = copy.deepcopy(kpis)
        detect_metrics = {kpi.metric: kpi for kpi in detect_kpis}
        ts_scores = []
        for kpi in kpis:
            ts_scores.extend(self.cal_metric_ab_score(kpi.metric, machine_id))

        ts_scores.sort(key=lambda x: x[1], reverse=True)
        ts_scores = ts_scores[:1]

        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts.metric,
                labels=_ts.labels,
                score=_score,
                entity_name=detect_metrics[_ts.metric.split("@")[0]].entity_name,
            )
            for _ts, _score in ts_scores
        ]

        return anomalies

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

    def get_training_data(self, start: datetime, end: datetime, metrics: List[str], machine_id: str)\
            -> Union[DataFrame, None]:
        """Get the training data to support model training"""
        logger.info('Get training data from %s to %s on %s!',
                    start.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    end.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    machine_id)
        _, x_dfs = self.get_dataframe(start, end, metrics, machine_ids=[machine_id])

        if not x_dfs:
            return None
        else:
            return x_dfs[0]

    def get_inference_data(self, start: datetime, end: datetime, metrics: List[str], machine_id: str)\
            -> List[Tuple[str, DataFrame]]:
        """Get data for the model inference and prediction"""
        logger.info('Get inference data during %s to %s on %s!',
                    start.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    end.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    machine_id)
        ids, x_dfs = self.get_dataframe(start, end, metrics, machine_ids=[machine_id])

        return list(zip(ids, x_dfs))

    def get_dataframe(self, start, end, metrics, machine_ids):
        """Gets the features during a period seperated by machine ids"""
        dataframes = []
        for machine_id in machine_ids:
            metric_dfs = []
            for metric in metrics:
                ts_list = self.data_loader.get_metric(
                    start, end, metric, operator='avg',
                    keys="machine_id", machine_id=machine_id)

                if len(ts_list) > 1:
                    raise ValueError(f'Got multiple time_series based'
                                     f'on machine id: {len(ts_list)}')

                if ts_list:
                    ts_list = ts_list[0]
                else:
                    ts_list = TimeSeries(metric, {}, [], [])

                metric_dfs.append(ts_list.to_df())

            df = reduce(lambda left, right: pd.DataFrame(left).join(right, how='outer'), metric_dfs)
            df = df.fillna(0)
            dataframes.append(df)

        return machine_ids, dataframes

    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) -> List[Anomaly]:
        """Multiple processes detect anomalies on individual machines"""
        logger.info(f'Execute model: {self.__class__.__name__}!')
        min_minutes = self.online_model.get_min_predict_minutes()
        start, end = dt.last(minutes=min_minutes)
        metrics = [k.metric for k in kpis]
        machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
        if not machine_ids:
            logger.warning('Empty machine_id, RETURN!')

        anomalies = []
        for _id in machine_ids:
            anomalies.extend(self.detect_machine_kpis(kpis, features, _id))

        return anomalies
