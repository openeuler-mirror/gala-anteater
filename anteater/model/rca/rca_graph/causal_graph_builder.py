# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：causal_graph_builder.py
Author: h00568282/huangbin 
Create Date: 2024/1/2 15:20
Notes:

"""

import warnings
import pandas as pd
import networkx as nx
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
warnings.filterwarnings("ignore")
from anteater.utils.log import logger


class CausalGraphBuilder:
    """
    利用元图构建基于指标的结构图
    """

    def __init__(self, meta_gaph: nx.DiGraph, data: pd.DataFrame, data_score: dict, front_end_metric: str,
                 machine_id) -> None:
        self._meta_graph = meta_gaph
        self._data = data
        self._data_score = data_score
        self._front_end_metric = front_end_metric
        self._machine_id = machine_id
        self._metric_list = None
        self._structure_graph = None
        self.virtual_node = "virtual_node"
        self.fv_metric_with_machine = self.get_machine_fv_metric()

    def load_data(self, args, special_sli_metrics):
        topk = args.get("topk", 10)
        maxlag = args.get("maxlag", 2)
        p_threshold = args.get("p_threshold", 0.05)

        # metric指标写入node点
        self.plug_meta_graph(special_sli_metrics, topk)

        # 根据元图做对应指标间的因果检验，构建具有因果关系的结构图
        self.create_structure_graph(maxlag=maxlag, p_threshold=p_threshold)

        return self._structure_graph

    @property
    def causal_graph(self):
        return self._structure_graph

    def plug_virtual_node(self):
        self._meta_graph.add_node(self.virtual_node)
        self._meta_graph.add_edge(self._machine_id, self.virtual_node)

    def build_entity_mapping(self):
        entity_metrics_mapping = {}
        for node in self._meta_graph.nodes():
            entity_metrics_mapping[node] = {"mapping_metrics": []}

        return entity_metrics_mapping

    def get_machine_fv_metric(self):
        return self._front_end_metric + '*' + self._machine_id

    def plug_meta_graph(self, special_sli_metrics: list, topk: int = 10) -> None:
        """
        元图的指标插装
        """
        # 异常分数从大到小排序
        sorted_score = sorted(self._data_score.items(), key=lambda x: -x[1])
        self._metric_list = [col for col,
        _ in sorted_score if col in self._data.columns]
        # self.plug_virtual_node()
        entity_metrics_mapping = self.build_entity_mapping()
        for metric in self._metric_list:
            entity = metric.split('*')[1]
            if entity not in list(self._meta_graph.nodes()):
                continue
            if "virtual_node" in entity_metrics_mapping.keys() and entity == self._machine_id and self._front_end_metric in metric:
                entity_metrics_mapping.get("virtual_node").get("mapping_metrics").append(metric)
                continue
            if "gala_gopher_proc_flow_perf_rx_delay" in metric:
                continue
            if len(entity_metrics_mapping.get(entity).get("mapping_metrics")) < topk:
                entity_metrics_mapping.get(entity).get(
                    "mapping_metrics").append(metric)

        # 保证一定有前端指标
        if "virtual_node" in entity_metrics_mapping.keys():
            if self.fv_metric_with_machine not in entity_metrics_mapping.get("virtual_node").get("mapping_metrics"):
                entity_metrics_mapping.get("virtual_node").get("mapping_metrics").append(self.fv_metric_with_machine)
        else:
            if self.fv_metric_with_machine not in entity_metrics_mapping.get(self._machine_id).get("mapping_metrics"):
                entity_metrics_mapping.get(self._machine_id).get("mapping_metrics").append(self.fv_metric_with_machine)

        # logger.info(f'add all front end metric to all nodes...')
        # for machine_id, front_end_metrics in self.all_front_end_metrics.items():
        #     # 当前前端指标节点跳过, 其他node的前端指标加入
        #     if machine_id not in entity_metrics_mapping.keys() or machine_id == self._machine_id:
        #         continue
        #     for front_end_metric in front_end_metrics:
        #         machine_front_end_metric = front_end_metric + '*' + machine_id
        #         if machine_front_end_metric not in entity_metrics_mapping.get(machine_id).get("mapping_metrics"):
        #             entity_metrics_mapping.get(machine_id).get("mapping_metrics").append(machine_front_end_metric)

        # self.print_entity_metrics_mapping(entity_metrics_mapping)
        nx.set_node_attributes(self._meta_graph, entity_metrics_mapping)
        logger.debug(f"元图的边: {self._meta_graph.edges()}")

    def print_entity_metrics_mapping(self, entity_metrics_mapping):
        for entity, entity_info in entity_metrics_mapping.items():
            mapping_metrics = entity_info.get("mapping_metrics")
            logger.info(f"entity: {entity}, entity_metrics_mapping: {len(mapping_metrics)}, {mapping_metrics}")

    def create_structure_graph(self, maxlag: int = 2, p_threshold: int = 0.05) -> None:
        """
        根据元图做对应指标间的因果检验，构建具有因果关系的结构图，
        """
        self._structure_graph = nx.DiGraph()
        for src_node, tar_node in self._meta_graph.edges():
            # 处理节点间无metric连接，fvmetric记录跟踪最低的p_value值连接，
            any_edge_flag = False
            causality_test_info = {
                'edge': [],
                'min_p_value': 100,
                'min_edge': None,
                "backup_edge": []
            }
            logger.info(f"causal test, src_node: {src_node}, tar_node:{tar_node} ********************")
            for src_metric in self._meta_graph.nodes.get(src_node).get("mapping_metrics"):
                for tar_metric in self._meta_graph.nodes.get(tar_node).get("mapping_metrics"):
                    try:
                        test_result = grangercausalitytests(
                            self._data[[tar_metric, src_metric]], maxlag=maxlag, verbose=False)
                    except Exception:
                        logger.error(f"grangercausalitytests：{tar_metric}, {src_metric}.")
                        if tar_metric == self.fv_metric_with_machine and (src_metric, tar_metric) not in \
                                causality_test_info['backup_edge']:
                            causality_test_info['backup_edge'].append((src_metric, tar_metric))
                        continue

                    mark = True
                    for _, result in test_result.items():
                        cur_p_value = result[0].get("ssr_ftest")[1]  # 显著性水平
                        # FIXME disk指标和sli强链接
                        # if "disk" in src_metric:
                        #     continue
                        if tar_metric == self.fv_metric_with_machine:
                            if causality_test_info['min_p_value'] > cur_p_value:
                                causality_test_info['min_p_value'] = cur_p_value
                                causality_test_info['min_edge'] = (src_metric, tar_metric)

                            if cur_p_value <= p_threshold:
                                if (src_metric, tar_metric) not in causality_test_info['edge']:
                                    causality_test_info['edge'].append((src_metric, tar_metric))

                        if cur_p_value > p_threshold:
                            mark = False

                    if mark:
                        if tar_metric == self.fv_metric_with_machine:
                            any_edge_flag = True
                        self._structure_graph.add_edge(src_metric, tar_metric)
                    # 添加孤立节点
                    elif src_metric not in self._structure_graph.nodes():
                        self._structure_graph.add_node(src_metric)
                    elif tar_metric not in self._structure_graph.nodes():
                        self._structure_graph.add_node(tar_metric)

            if self._machine_id == tar_node and not any_edge_flag:
                if causality_test_info['edge']:
                    self._structure_graph.add_edges_from(causality_test_info['edge'])
                elif causality_test_info['min_edge']:
                    self._structure_graph.add_edge(*causality_test_info['min_edge'])
                else:
                    # 任一添加一条边fv_pod_metric to next_pod_metric
                    backup_edge_len = len(causality_test_info['backup_edge'])
                    if backup_edge_len > 0:
                        random_index = np.random.randint(0, backup_edge_len)
                        self._structure_graph.add_edge(*(causality_test_info['backup_edge'][random_index]))

        # 为结构图上的指标节点添加值属性
        for node in self._structure_graph.nodes():
            nx.set_node_attributes(self._structure_graph, {
                node: {"timelist": self._data[node].tolist(),
                       "anomaly_score": self._data_score.get(node, 0)}})

        # 保证结构图连通，提取子图
        self._extract_subgraph()
        is_valid_graph = self.is_valid_graph()

        if not is_valid_graph:
            self._structure_graph = nx.DiGraph()
        logger.info(f"causal structure_graph edges: {len(self._structure_graph.edges())} ********************")

    def _extract_subgraph(self):
        ''' 提取包含fv_metric的子图 '''
        if not len(self._structure_graph.nodes()):
            return
        if not nx.is_weakly_connected(self._structure_graph):
            # 获取弱连通分量
            for component in nx.weakly_connected_components(self._structure_graph):
                if self._front_end_metric + '*' + self._machine_id in list(component):
                    self._structure_graph = self._structure_graph.subgraph(component)
                    break

    def is_valid_graph(self):
        ''' 如果子图不包含fv_machine, 则该子图无效 '''
        is_valid_graph = False
        for src, target in self._structure_graph.edges():
            if self._machine_id in src or self._machine_id in target:
                is_valid_graph = True
                break

        return is_valid_graph


def main():
    pass


if __name__ == "__main__":
    main()
