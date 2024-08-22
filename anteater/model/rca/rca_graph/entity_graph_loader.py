# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：entity_graph_loader.py
Author: h00568282/huangbin 
Create Date: 2024/1/2 15:18
Notes:

"""

import networkx as nx

from anteater.utils.log import logger
from anteater.model.rca.rca_graph.prune_bidirected_graph import PrunedMetaGraph


class EntityGraphLoader():
    def __init__(self, fv_machine: str, entity_topo_config, machine_anomaly_scores, anomaly_result, entity_topo_dict):
        self.fv_machine = fv_machine
        self.entity_topo_config = entity_topo_config
        self.machine_anomaly_scores = machine_anomaly_scores
        self.anomaly_result = anomaly_result
        self.entity_topo_dict = entity_topo_dict

    def _init_entity_graph(self, entity_topo_config: dict) -> nx.DiGraph:
        """
        构建元图
        """
        entity_topo_config = self.entity_topo_dict
        meta_graph = nx.DiGraph()
        # 添加节点
        for node in entity_topo_config:
            meta_graph.add_node(node)
        # 添加有向边
        for node, neighbors in entity_topo_config.items():
            for neighbor in neighbors:
                meta_graph.add_edge(node, neighbor)

        return meta_graph

    def load_data(self):
        '''
            从json文件中初始化meta graph; 对meta graph进行裁剪：双向访问裁剪，fv_machine下游machine连接关系处理
        '''

        entity_graph = self._init_entity_graph(self.entity_topo_config)

        # 裁剪元图
        logger.info(f"machine id: {self.fv_machine}, before pruning, meta graph edges: {len(entity_graph.edges())}")

        graph_pruner = PrunedMetaGraph(entity_graph, self.machine_anomaly_scores)
        pruned_entity_graph = graph_pruner.run(self.fv_machine)

        logger.info(
            f"machine id: {self.fv_machine}, after pruning, meta graph edges: {len(pruned_entity_graph.edges())}")

        return pruned_entity_graph


def main():
    pass


if __name__ == "__main__":
    main()
