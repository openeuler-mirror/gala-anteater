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
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from anteater.core.kpi import JobConfig
from anteater.source.anomaly_report import AnomalyReport

from anteater.utils.log import logger
from anteater.config import ArangodbConf
from anteater.model.rca.data_load.get_data import get_data
from anteater.model.rca.root_cause_infer import RootCauseInfer
from anteater.model.rca.template.kafka import KafkaProvider
from anteater.model.rca.cause_inference.arangodb import connect_to_arangodb, query_recent_topo_ts, query_all


class RCA:
    config_file = None

    def __init__(self, provider: KafkaProvider, reporter: AnomalyReport, job_config: JobConfig, arangodb: ArangodbConf):
        self.provider = provider
        self.reporter = reporter
        self.job_config = job_config.model_config
        self.init_time = datetime.now(timezone.utc).astimezone().astimezone() - timedelta(hours=12)
        self.arangodb_config = arangodb

        self.entity_topo_dict = {}

        self.last_event_id = []

    @staticmethod
    def extract_pod_id(label):
        pod_id = label.split("_pod_")[-1]
        return pod_id

    def get_raw_graph_from_arangodb(self, anomaly_result):
        url = self.arangodb_config.url
        db_name = self.arangodb_config.db_name
        arango_db = connect_to_arangodb(url, db_name)
        recent_topo_ts = query_recent_topo_ts(arango_db, anomaly_result['Timestamp'] // 1000)
        aql_query = f"for e in connect filter e.timestamp == {recent_topo_ts} return e"

        topo_msg = query_all(arango_db, aql_query)
        logger.info(f"[INFO] topo msg: {len(topo_msg)}: {topo_msg}")

        entity_topo_dict = {}
        for edge_msg in topo_msg:
            from_node = self.extract_pod_id(edge_msg["_from"])
            to_node = self.extract_pod_id(edge_msg["_to"])
            if from_node and to_node:
                if to_node in entity_topo_dict.keys():
                    connected_nodes = entity_topo_dict[to_node]
                    if from_node not in connected_nodes:
                        connected_nodes.append(from_node)
                    else:
                        logger.info(f"[INFO] {from_node} has been in topo graph!")

                else:
                    entity_topo_dict[to_node] = [from_node]

        logger.info(f"[INFO] Extract raw entity topo: {entity_topo_dict}")

        return entity_topo_dict

    @staticmethod
    def get_metrics(data_dir, machine_id, anomaly_result, end_timestamp, kpis):
        data_path = os.path.join(data_dir, machine_id)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        start = datetime.fromtimestamp(end_timestamp).astimezone() - timedelta(minutes=10)
        end = datetime.fromtimestamp(end_timestamp).astimezone()
        metrics_df = get_data(start, end, kpis, machine_id)
        metrics_df.to_csv(os.path.join(data_path, 'metric.csv'), index=False)
        with open(os.path.join(data_path, 'result.json'), mode="w") as json_file:
            json.dump([anomaly_result], json_file, indent=4)

    @staticmethod
    def get_single_machine_metrics(machine_id, end_timestamp, kpis):
        start = datetime.fromtimestamp(end_timestamp).astimezone() - timedelta(minutes=10)
        end = datetime.fromtimestamp(end_timestamp).astimezone()
        metrics_df = get_data(start, end, kpis, machine_id)
        return metrics_df

    def debug_trigger_rca(self):
        anomaly_data = []
        machine_ids = []
        all_machines_df = {}
        anomaly_data_dict = {}
        data_dir = "/data/hbdir/root_cause_analysis_v1/dataset"
        for path in os.listdir(data_dir):
            if path.endswith(".json") or path.endswith("html") or path.endswith("png"):
                continue

            data_path = os.path.join(data_dir, path)
            metrics_df = pd.read_csv(os.path.join(data_path, 'metric.csv'))
            with open(os.path.join(data_path, 'result.json'), 'r') as f:
                anomaly_result = json.load(f)
                anomaly_result = anomaly_result[0]
            all_machines_df[path] = metrics_df
            anomaly_data_dict[path] = anomaly_result
            anomaly_data.append(anomaly_result)
            machine_ids.append(path)

        for index, (machine_id, anomaly_result) in enumerate(zip(machine_ids, anomaly_data)):
            logger.info(f"machine_id: {machine_id}")
            tmp_attr = anomaly_result["Attributes"]

            is_anomaly = anomaly_result['is_anomaly']
            if is_anomaly:
                logger.info(
                    f'{index}-th cur machine id, {machine_id}, {tmp_attr}， is_anomaly: {is_anomaly}***************')
            if anomaly_result['is_anomaly']:
                root_cause_infer = RootCauseInfer(self.job_config, anomaly_result, anomaly_data_dict, machine_ids,
                                                  all_machines_df, self.arangodb_config)
                res = root_cause_infer.excute(seed=100)
                self.provider.send_rca_message(res)

                logger.info(f"[Info] RCA End, final output message: {res}.")

    def trigger_rca(self, anomaly_data):
        metric_candidates = self.job_config["metrics"]
        self.last_event_id = self.last_event_id[-100:]
        if len(anomaly_data):
            for i in range(len(anomaly_data)):
                machine_ids = []
                anomaly_results = {}
                all_machines_metric_df = {}
                if not anomaly_data[i].get("is_anomaly", False):
                    continue
                event_id = anomaly_data[i]["Attributes"]["event_id"]
                if event_id in self.last_event_id:
                    logger.info(f"************ i={i}, event_id={event_id} has detected.")
                    continue

                logger.info(f'********* i={i}, {anomaly_data[i]["Attributes"]}.')
                self.last_event_id.append(event_id)
                cur_timestamp = anomaly_data[i]["Timestamp"]
                cur_datetime = datetime.fromtimestamp(cur_timestamp // 1000)
                machine_id = anomaly_data[i]["Attributes"]["event_id"].split("_")[1]
                metrics_df = self.get_single_machine_metrics(machine_id, cur_timestamp // 1000, metric_candidates)
                all_machines_metric_df[machine_id] = metrics_df
                all_anomaly_datas = self.provider.range_query(cur_datetime - timedelta(minutes=5),
                                                              cur_datetime + timedelta(minutes=5))  # minutes=5

                for _anomaly_data in all_anomaly_datas:
                    _machine_id = _anomaly_data["Attributes"]["event_id"].split("_")[1]
                    metrics_df = self.get_single_machine_metrics(machine_id, cur_timestamp // 1000, metric_candidates)
                    all_machines_metric_df[_machine_id] = metrics_df
                    anomaly_results[_machine_id] = _anomaly_data
                    machine_ids.append(_machine_id)
                try:
                    root_cause_infer = RootCauseInfer(self.job_config, anomaly_data[i], anomaly_results, machine_ids,
                                                      all_machines_metric_df, self.entity_topo_dict)
                    res = root_cause_infer.excute(seed=100)
                    logger.info(f"[Info] RCA End, final output message: {res}.")
                    self.provider.send_rca_message(res)
                except Exception:
                    logger.error(f"Anomaly data has wrong format. Please check!")

    def execute(self):
        logger.info(f'Run rca model: {self.__class__.__name__}!')
        now_time = datetime.now(timezone.utc).astimezone().astimezone()
        try:
            anomaly_data = self.provider.range_query(now_time - timedelta(minutes=10), now_time)
        except Exception:
            anomaly_data = []

        logger.info(f"[Info] Current time: {now_time} has anomaly data: {len(anomaly_data)}.")
        if len(anomaly_data) < 1:
            self.init_time = now_time
            return

        self.entity_topo_dict = self.get_raw_graph_from_arangodb(anomaly_data[0])
        self.trigger_rca(anomaly_data)

        self.init_time = now_time
