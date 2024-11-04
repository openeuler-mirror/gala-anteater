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

from typing import List

from anteater.model.rca.rca_entry import RCA
from anteater.provider.kafka import KafkaProvider
from anteater.source.anomaly_report import AnomalyReport
from anteater.utils.data_load import load_jobs
from anteater.utils.log import logger
from anteater.config import ArangodbConf

class RootCauseAnalysis:
    """The root cause analysis class"""

    def __init__(self, provider: KafkaProvider, reporter: AnomalyReport, arangodb: ArangodbConf):
        """The root cause analysis initializer"""
        self.provider = provider
        self.reporter = reporter

        self.rca = self.load_rca(arangodb)

    def load_rca(self, arangodb) -> RCA:
        """load rca from config file"""
        rca = None
        for job_config in load_jobs():
            if job_config.enable and job_config.job_type == 'root_cause_analysis':
                rca = RCA(self.provider, self.reporter, job_config, arangodb)

        job_count = 1 if rca else 0

        logger.info('Root Cause loaded %d jobs.', job_count)

        return rca

    def run(self):
        """Executes root cause analysis task"""
        logger.info('START: A new gala-anteater rca task is scheduling!')

        if self.rca:
            self.rca.execute()

        logger.info('END!: rca task')
