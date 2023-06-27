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

import copy
import pandas as pd

from pandas import DataFrame
from datetime import datetime, timedelta
from functools import reduce
from multiprocessing import Pool
from typing import List, Tuple

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI, ModelConfig, Feature
from anteater.core.time_series import TimeSeries
from anteater.feature_extraction.feature_util import select_relevant_kpi
from anteater.model.algorithms.slope import check_trend
from anteater.model.detector.base import Detector
from anteater.model.online_model import OnlineModel
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class OnlineDetector(Detector):
    """The anomaly detector base class"""

    def __init__(self, data_loader: MetricLoader, conf: ModelConfig):
        """The detector base class initializer"""
        super().__init__(data_loader)
        self.online_model = OnlineModel(conf) if (conf and conf.enable) else None
        self.config = conf
        self.anomaly_scores = None
        self.cause_list = None
        self.candidates = {}
        self.machine_training = {}

    def detect_machine_kpis(self, kpis: List[KPI], features: List[Feature], machine_id, cause_top=50):
        """Executes anomaly detection on kpis"""
        sli_metrics = [kpi.metric for kpi in kpis]
        detect_kpis = copy.deepcopy(kpis)
        detect_kpis.extend(features)
        detect_metrics = [kpi.metric for kpi in detect_kpis]
        detect_metrics.sort()
        anomalies = []
        start, end = dt.last(minutes=self.online_model.min_predict_minutes)
        for machine_id, x_df in self.get_inference_data(start, end, detect_kpis, machine_id):
            self.online_model.get_machine_id(machine_id)
            y_pred, sli_pred, abnormal_sli_metric = self.online_model.predict(
                x_df, machine_id, detect_metrics, sli_metrics)
            if len(y_pred) == 0:
                if machine_id in self.machine_training:
                    return anomalies
                else:
                    self.machine_training[machine_id] = True
                train_df = self.get_training_data(start - timedelta(hours=24), end, detect_kpis, machine_id)
                if len(train_df) > 0:
                    self.online_model.training(train_df, machine_id, detect_metrics, sli_metrics)
                return anomalies
            if machine_id not in self.machine_training:
                self.machine_training[machine_id] = True
            if self.online_model.is_abnormal(y_pred) or self.online_model.is_abnormal(sli_pred):
                pearson_record = select_relevant_kpi(x_df, x_df[abnormal_sli_metric])
                cause_list, now_time, anomaly_scores_list = self.online_model.trigger_locate(machine_id, pearson_record)
                if len(cause_list) == 0 or len(anomaly_scores_list) == 0:
                    return anomalies
                self.anomaly_scores = copy.deepcopy(anomaly_scores_list[0])
                self.cause_list = copy.deepcopy(cause_list[0])
                anomalies.extend(self.find_abnormal_kpis(machine_id, kpis, features, anomaly_scores_list[0], top_n=1))
                new_features = copy.deepcopy(features)
                candidates = [c["metric"] for c in cause_list[0]["Resource"]["cause_metrics"][:cause_top]]
                remove_features = [f for f in new_features if f.metric not in candidates]
                for r in remove_features:
                    new_features.remove(r)
                anomalies = self.get_usad_root_causes(anomalies, new_features, self.anomaly_scores)
                anomalies = self.cal_usad_kpi_anomaly_score(anomalies, self.anomaly_scores)

        return anomalies

    def find_abnormal_kpis(self, machine_id: str, kpis: List[KPI], features: List[Feature], scores, top_n=1):
        """Find abnormal kpis when detected anomaly events"""
        sli_metrics = [kpi.metric for kpi in kpis]
        detect_kpis = copy.deepcopy(kpis)
        detect_metrics = {kpi.metric: kpi for kpi in detect_kpis}

        ts_scores = []
        for kpi in kpis:
            tmp_ts_scores, metric_similar = self.cal_usad_anomaly_score(
                kpi.metric, kpi.description, machine_id, float(scores[kpi.metric]), top_n=1)
            if tmp_ts_scores:
                for _ts_score in tmp_ts_scores:
                    _ts_score.score = _ts_score.score if check_trend(_ts_score.ts.values, kpi.atrend) else 0
                ts_scores.extend(tmp_ts_scores)

        ts_scores = [v for v in ts_scores if v.score > 0]
        ts_scores = sorted(ts_scores, key=lambda x: x.score, reverse=True)
        ts_scores = ts_scores[: top_n]

        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts_score.ts.metric,
                labels=_ts_score.ts.labels,
                score=_ts_score.score,
                entity_name=detect_metrics[_ts_score.ts.metric.split(
                    "@")[0]].entity_name,
                description=detect_metrics[_ts_score.ts.metric.split(
                    "@")[0]].description
            )
            for _ts_score in ts_scores
        ]

        return anomalies

    def get_training_data(self, start: datetime, end: datetime, kpis: List[KPI], machine_id)\
            -> DataFrame:
        """Get the training data to support model training"""
        logger.info(
            f"Get training data during {start} to {end} on {machine_id}!")
        ids, x_dfs = self.get_dataframe(
            start, end, kpis, machine_ids=[machine_id])
        logger.info("The shape of training data")
        if not x_dfs:
            del self.machine_training[machine_id]
            return []

        return x_dfs[0]

    def get_inference_data(self, start: datetime, end: datetime, kpis: List[KPI], machine_id=None)\
            -> List[Tuple[str, DataFrame]]:
        """Get data for the model inference and prediction"""
        if machine_id is None:
            ids, x_dfs = self.get_dataframe(start, end, kpis)
        else:
            ids, x_dfs = self.get_dataframe(
                start, end, kpis, machine_ids=[machine_id])

        return list(zip(ids, x_dfs))

    def get_dataframe(self, start, end, kpis, machine_ids=None, interval=5):
        """Gets the features during a period seperated by machine ids"""
        metrics = [kpi.metric for kpi in kpis]

        if machine_ids is None:
            machine_ids = self.get_unique_machine_id(start, end, kpis)
            if not machine_ids:
                logger.warning(
                    "Cannot get unique machine ids from Prometheus!")
        else:
            machine_ids = machine_ids

        dataframes = []
        for machine_id in machine_ids:
            metric_records = {}
            metric_dfs = []
            metrics_list = []
            for metric in metrics:
                _ts_list = self.data_loader.\
                    get_metric(start, end, metric, operator='avg',
                               keys="machine_id", machine_id=machine_id)

                if len(_ts_list) > 1:
                    raise ValueError(
                        f'Got multiple time_series based on machine id: {len(_ts_list)}')

                if _ts_list:
                    _ts_list = _ts_list[0]
                else:
                    _ts_list = TimeSeries(metric, {}, [], [])
                if _ts_list.to_df().columns[0] not in metrics_list:
                    metrics_list.append(_ts_list.to_df().columns[0])
                    metric_dfs.append(_ts_list.to_df())

            df = reduce(lambda left, right: pd.DataFrame(
                left).join(right, how='outer'), metric_dfs)
            df = df.fillna(0)
            df["timestamp"] = df.index.to_list()
            dataframes.append(df)

        return machine_ids, dataframes

    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) -> List[Anomaly]:
        """Multiple processes detect anomalies on individual machines"""
        logger.info(f"Execute model: {self.__class__.__name__}!")
        start, end = dt.last(minutes=self.online_model.min_predict_minutes)
        machine_ids = self.get_unique_machine_id(start, end, kpis)

        p = Pool(processes=len(machine_ids))
        anomalies = []
        result = []
        for _id in machine_ids:
            result.append(p.apply_async(self.detect_machine_kpis, args=(kpis, features, _id)))
        p.close()
        p.join()
        for r in result:
            anomalies.extend(r.get())

        return anomalies
