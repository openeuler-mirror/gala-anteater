# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：root_cause_infer.py
Author: h00568282/huangbin 
Create Date: 2023/12/29 15:59
Notes:

"""
import os
import networkx as nx
import numpy as np
import pandas as pd

from anteater.model.rca.template import RcaReporter
from anteater.model.rca.data_load import RcaMetricLoader
from anteater.model.rca.rca_graph import EntityGraphLoader, CausalGraphBuilder
from anteater.model.rca.root_cause_infer.prob_matrix_builder import ProbMatrixBuilder
from anteater.model.rca.root_cause_infer.random_walk import RandomWalker
from anteater.model.rca.root_cause_infer.path_builder import MinPathSearcher
from anteater.utils.log import logger


class RootCauseInfer():
    def __init__(self, config, anomaly_result, all_anomaly_results, machine_ids, all_machines_df,
                 entity_topo_dict) -> None:
        self._config = config
        self.entity_topo_dict = entity_topo_dict
        self.args = self._config.get("args")
        self._data_dir = self._config.get("args").get("data_dir")
        self.special_sli_metrics = self._config.get("special_sli_metrics")
        self.metric_candidates = self._config.get("metrics")

        logger.info(f"===================== dataset: {self._data_dir} =====================")
        self._filename = os.path.basename(self._data_dir)
        self.anomaly_results = anomaly_result
        self.all_anomaly_results = all_anomaly_results
        self.machine_ids = machine_ids

        self.machine_id = self.parse_machine_id(anomaly_result)
        self.front_end_metric = self.parse_front_end_metric(anomaly_result)
        self.metric_to_entity = {}
        self._output_dir = self._config.get("args").get("output_dir")
        self.mk_output_dir()
        self.all_pods = {}
        self.pod_states = {}
        self.pod_name = {}
        self.pod_instance = {}
        self.pod_job = {}

        # 加载metric序列和异常分数
        self.all_metrics_df, self.all_metric_top, self.machine_anomaly_scores, self.all_front_end_metrics = self.load_metrics_ts_and_scores(
            all_machines_df)
        # 生成entity graph和causal graph
        self.entity_graph, self.causal_graph = self.generate_graphs()

        # 初始化pods信息
        self.generate_pod_infos()

    def mk_output_dir(self):
        if self._output_dir is None:
            self._output_dir = os.path.join(os.path.dirname(os.path.dirname(self._data_dir)),
                                            "output",
                                            self._filename)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def generate_graphs(self):
        # 原始拓扑图构建和预处理
        entity_graph_loader = EntityGraphLoader(self.machine_id, self._config.get("meta_graph"),
                                                self.machine_anomaly_scores, self.anomaly_results, self.entity_topo_dict)
        entity_graph = entity_graph_loader.load_data()
        causal_graph_builder = CausalGraphBuilder(entity_graph, self.all_metrics_df, self.all_metric_top,
                                                  self.front_end_metric,
                                                  self.machine_id)
        causal_graph = causal_graph_builder.load_data(self.args, self.metric_candidates)

        return entity_graph, causal_graph

    def load_metrics_ts_and_scores(self, all_machines_df):
        metric_loader = RcaMetricLoader(self.args, self.metric_candidates, self.special_sli_metrics)
        all_metrics_ts, all_metric_scores, machine_anomaly_scores, all_front_end_metrics = metric_loader.load_data(
            self.machine_ids, all_machines_df, self.all_anomaly_results)

        return all_metrics_ts, all_metric_scores, machine_anomaly_scores, all_front_end_metrics

    @staticmethod
    def parse_machine_id(anomaly_result):
        return anomaly_result["Attributes"]["event_id"].split("_")[1]

    @staticmethod
    def parse_front_end_metric(anomaly_result):
        if "@" in anomaly_result['Resource']['metric']:
            front_end_metric = anomaly_result['Resource']['metric'].split('@')[0]
        else:
            front_end_metric = anomaly_result['Resource']['metric']

        return front_end_metric

    def get_pod_state(self):
        for anomaly_result in self.all_anomaly_results.values():
            pod_id = anomaly_result['Attributes']['event_id'].split("_")[1]
            if pod_id in self.all_pods:
                self.all_pods[pod_id] = anomaly_result['Resource']['metric'].split("@")[0]
                if anomaly_result['is_anomaly']:
                    self.pod_states[pod_id] = 'abnormal'
                else:
                    self.pod_states[pod_id] = 'normal'
                self.pod_name[pod_id] = anomaly_result['Attributes']["cause_metric"]['labels']['pod']
                self.pod_instance[pod_id] = anomaly_result['Attributes']["cause_metric"]['labels']['instance']
                self.pod_job[pod_id] = anomaly_result['Attributes']["cause_metric"]['labels']['job']

    def get_pod_mapping_sli_metric(self):
        for anomaly_result in self.all_anomaly_results.values():
            pod_id = anomaly_result['Attributes']['event_id'].split("_")[1]
            if pod_id in self.all_pods:
                self.all_pods[pod_id] = anomaly_result['Resource']['metric'].split("@")[0]

    def generate_pod_infos(self):
        meta_graph_config = self.entity_topo_dict
        for src_pod in meta_graph_config.keys():
            if src_pod not in self.all_pods:
                self.all_pods[src_pod] = ""
                self.pod_states[src_pod] = False
            for tar_pod in meta_graph_config[src_pod]:
                if tar_pod not in self.all_pods:
                    self.all_pods[tar_pod] = ""
                    self.pod_states[tar_pod] = False

        self.get_pod_mapping_sli_metric()
        self.get_pod_state()
        # logger.info(f"self.all_pods: {self.all_pods}")
        for node in self.entity_graph.nodes():
            sli_metric = self.all_pods[node]
            nx.set_node_attributes(self.entity_graph, {
                node: {"timelist": self.all_metrics_df[sli_metric + "*" + node].tolist(),
                       "anomaly_score": self.all_metric_top.get(sli_metric + "*" + node, 0)}})

    def excute(self, seed=100):
        """
            运行根因定位算法核心
        """
        error_counts = 0
        outputs = []
        anomaly_results = [self.anomaly_results]
        args = self._config.get("args")

        # 固定随机游走的随机概率
        np.random.seed(seed)
        random_walk_prob = np.random.uniform(0, 1, args.get("num_loop"))
        prob_matrix_builder = ProbMatrixBuilder(args)
        random_walker = RandomWalker(args, random_walk_prob)

        front_end_metric = self.front_end_metric + "*" + self.machine_id

        index = 0
        graph = self.causal_graph

        output_dict = {}
        time_stamp = anomaly_results[index].get("Timestamp")
        output_dict["Timestamp"] = time_stamp
        output_dict["TimeStr"] = str(pd.to_datetime(anomaly_results[index].get("Timestamp") // 1000, unit="s"))
        output_dict["front_end_metric"] = front_end_metric

        # 概率矩阵
        transfer_matrix = prob_matrix_builder(graph, front_end_metric)

        # 随机游走
        rw_result, location_result = random_walker(graph, transfer_matrix, front_end_metric)
        output_dict["random_walk_result"] = rw_result

        # 搜索路径
        path_searcher = MinPathSearcher(args, time_stamp, self.entity_graph, self.machine_anomaly_scores)
        path_result, error_count = path_searcher(location_result, self.machine_id)
        output_dict.update(path_result)
        # 结果上报
        rca_reporter = RcaReporter(output_dict, self.anomaly_results, self.all_anomaly_results, self.pod_states,
                                   self.pod_name, self.pod_instance, self.pod_job)
        reporter_msg = rca_reporter(self.special_sli_metrics, self.all_front_end_metrics)

        outputs.append(output_dict)
        error_counts += error_count

        logger.info(f"error_counts: {error_counts}")

        return reporter_msg
