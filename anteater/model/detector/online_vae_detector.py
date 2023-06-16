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

from datetime import datetime
from functools import reduce
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI, ModelConfig
from anteater.core.time_series import TimeSeries
from anteater.model.algorithms.slope import check_trend
from anteater.model.online_vae_model import OnlineVAEModel
from anteater.model.detector.base import Detector
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class OnlineVAEDetector(Detector):
    """The online vae detector"""

    def __init__(self, data_loader: MetricLoader, conf: ModelConfig):
        """The detector base class initializer"""
        super().__init__(data_loader)
        self.online_model = OnlineVAEModel(conf) if (conf and conf.enable) else None

    def detect_kpis(self, kpis: List[KPI]):
        """Executes anomaly detection on kpis"""
        if self.online_model.need_retrain():
            start, end = dt.last(hours=self.online_model.min_retrain_hours)
            df = self.get_training_data(start, end, kpis)
            self.online_model.training(df)

        anomalies = []
        start, end = dt.last(minutes=self.online_model.min_predict_minutes)
        for machine_id, x_df in self.get_inference_data(start, end, kpis):
            y_pred = self.online_model.predict(x_df)
            if self.online_model.is_abnormal(y_pred):
                anomalies.extend(self.find_abnormal_kpis(machine_id, kpis, top_n=1))

        return anomalies

    def find_abnormal_kpis(self, machine_id: str, kpis: List[KPI], top_n=1):
        """Find abnormal kpis when detected anomaly events"""
        metric_kpi = {kpi.metric: kpi for kpi in kpis}
        ts_scores = []
        for kpi in kpis:
            tmp_ts_scores = self.cal_metric_ab_score(kpi.metric, machine_id)
            for _ts, _score in tmp_ts_scores:
                if not check_trend(_ts.values, kpi.atrend):
                    break
                ts_scores.append((_ts, _score))

        ts_scores = [t for t in ts_scores if t[1] > 0]
        ts_scores.sort(key=lambda x: x.score, reverse=True)
        ts_scores = ts_scores[: top_n]
        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts.metric,
                labels=_ts.labels,
                score=_score,
                entity_name=metric_kpi[_ts.metric].entity_name
            )
            for _ts, _score in ts_scores
        ]

        return anomalies

    def get_training_data(self, start: datetime, end: datetime, kpis: List[KPI])\
            -> DataFrame:
        """Get the training data to support model training"""
        logger.info(f"Get training data during {start} to {end}!")
        _, dfs = self.get_dataframe(start, end, kpis)
        if not dfs:
            return pd.DataFrame()

        x_df = reduce(lambda left, right: pd.concat([left, right], axis=0), dfs)
        logger.info("The shape of training data")

        return x_df

    def get_inference_data(self, start: datetime, end: datetime, kpis: List[KPI])\
            -> List[Tuple[str, DataFrame]]:
        """Get data for the model inference and prediction"""
        ids, x_dfs = self.get_dataframe(start, end, kpis)

        return list(zip(ids, x_dfs))

    def get_dataframe(self, start, end, kpis):
        """Gets the features during a period seperated by machine ids"""
        metrics = [kpi.metric for kpi in kpis]

        machine_ids = self.get_unique_machine_id(start, end, kpis)
        if not machine_ids:
            logger.warning("Cannot get unique machine ids from Prometheus!")

        dataframes = []
        for machine_id in machine_ids:
            metric_dfs = []
            for metric in metrics:
                _ts_list = self.data_loader.\
                    get_metric(start, end, metric, operator='avg', keys="machine_id", machine_id=machine_id)

                if len(_ts_list) > 1:
                    raise ValueError(f'Got multiple time_series based on machine id: {len(_ts_list)}')

                if _ts_list:
                    _ts_list = _ts_list[0]
                else:
                    _ts_list = TimeSeries(metric, {}, [], [])

                metric_dfs.append(_ts_list.to_df())

            df = reduce(lambda left, right: pd.DataFrame(left).join(right, how='outer'), metric_dfs)
            df = df.fillna(0)

            dataframes.append(df)

        return machine_ids, dataframes
