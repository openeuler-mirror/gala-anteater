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

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


class TimeSeries:
    """The time series class"""
    def __init__(
            self,
            metric: str,
            labels: Dict,
            time_stamps: List[int],
            values: List[float]):
        """The time series initializer"""
        self.metric = metric
        self.labels = labels.copy()
        self.time_stamps = time_stamps.copy()
        self.values = values.copy()

        self.__id = f'{metric}'

    def __eq__(self, other):
        """Returns if the 'self' equals to the 'other' or not."""
        labels1 = tuple(sorted([(k, v) for k, v in self.labels.items()]))
        labels2 = tuple(sorted([(k, v) for k, v in other.labels.items()]))

        return self.metric == other.metric and labels1 == labels2

    @property
    def id(self) -> str:
        """Get the unique id for this time series"""
        return self.__id

    def extend(self, time_stamps: List[int], values: List[float]) -> None:
        """Extends the time series data set by appending time stamps and it's values"""
        if len(time_stamps) != len(values):
            raise ValueError("Extend Error: different length between time_stamps and values!")
        self.time_stamps.extend(time_stamps)
        self.values.extend(values)

    def insert(self, other):
        if self != other:
            raise ValueError("InsertError: different 'TimeSeries' instances!")

        self_ts_values = list(zip(self.time_stamps, self.values))
        other_ts_values = list(zip(other.time_stamps, self.values))

        ts_values = self_ts_values + other_ts_values
        ts_values.sort()

        self.time_stamps = [v[0] for v in ts_values]
        self.values = [v[1] for v in ts_values]

    def to_df(self, name: str = None) -> pd.Series:
        """Convert the time series to the pandas DataFrame"""
        timestamp = pd.to_datetime(np.asarray(self.time_stamps).astype(float) * 1000, unit="ms")
        index = pd.to_datetime(timestamp)
        np_values = np.asarray(self.values)

        if not name:
            name = self.id

        series = pd.Series(np_values, index=index, name=name)
        series = series[~series.index.duplicated()]

        df = series.to_frame()

        return df


@dataclass
class TimeSeriesScore:
    ts: TimeSeries
    score: float
    description: str
