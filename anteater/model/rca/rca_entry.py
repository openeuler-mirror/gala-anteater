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

from datetime import datetime, timezone, timedelta

from anteater.core.kpi import JobConfig
from anteater.source.anomaly_report import AnomalyReport
from anteater.provider.kafka import KafkaProvider
from anteater.utils.log import logger


class RCA:
    config_file = None

    def __init__(self, provider: KafkaProvider,
                 reporter: AnomalyReport, job_config: JobConfig):
        self.provider = provider
        self.reporter = reporter
        self.job_config = job_config
        self.init_time = datetime.now(timezone.utc).astimezone().astimezone() - timedelta(days=1)

    def execute(self):
        logger.info(f'Run rca model: {self.__class__.__name__}!')

        now_time = datetime.now(timezone.utc).astimezone().astimezone()
        anomaly_data = self.provider.range_query(self.init_time, now_time)

        if len(anomaly_data) < 1:
            self.init_time = now_time
            return

        for i in range(len(anomaly_data)):
            if not anomaly_data[i]["is_anomaly"]:
                continue

            anomaly_datetime = datetime.fromtimestamp(anomaly_data[i]["Timestamp"] // 1000)
            all_anomaly_data = self.provider.range_query(anomaly_datetime - timedelta(minutes=5),
                                                         anomaly_datetime + timedelta(minutes=5))
            # 触发定界定位
            # trigger_rca(self.job_config, anomaly_data[i], all_anomaly_data, machine_ids)

        self.init_time = now_time

