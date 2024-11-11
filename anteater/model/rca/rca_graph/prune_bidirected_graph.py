# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：prune_bidirected_graph.py
Author: h00568282/huangbin 
Create Date: 2023/11/11 10:56
Notes:

"""
import os
import json
from typing import Dict
import networkx as nx
from anteater.model.rca.rca_graph.utils import create_meta_graph
from anteater.utils.log import logger


class PrunedMetaGraph:
    def __init__(self, meta_graph: nx.DiGraph, machine_anomaly_scores: Dict):
        self._meta_graph = meta_graph
        
        self.total_anomaly_scores = self._process_anomaly_scores(machine_anomaly_scores)
        self.bidirected_edges = self._find_bidirected_edges()
        self.searched_edges = []
        self.pruned_edges = []

    @staticmethod
    def _process_anomaly_scores(all_anomaly_scores):
        values_dict = dict()
        for key, raw_value in all_anomaly_scores.items():
            metric_len = len(raw_value)
            if metric_len:
                value = sum(raw_value.values())
            else:
                value = 0
            
            values_dict[key] = value
        
        values_dict = dict(sorted(values_dict.items(), key=lambda x: -x[1]))

        return values_dict

    def _find_bidirected_edges(self):
        bidirected_edges = []
        for src_node, tar_node in self._meta_graph.edges():
            if (tar_node, src_node) in self._meta_graph.edges():
                bidirected_edges.append((src_node, tar_node))

        return bidirected_edges

    def _get_edge_score(self, cur_edge):
        return self.total_anomaly_scores.get(cur_edge[0], 0)

    def _iter_prune(self, front_end_metric):
        selected_edges = []
        for edge in self._meta_graph.edges():
            if front_end_metric == edge[1]:
                if edge in self.searched_edges:
                    continue
                else:
                    selected_edges.append(edge)
                    self.searched_edges.append(edge)

        self._cache_pruned_biedges(selected_edges)

        # 23-12-11 插入选择节点的重要性, 异常分数越高排名越靠前，指向它的边保留
        sorted_selected_edges = []
        for cur_edge in selected_edges:
            cur_edge_score = self._get_edge_score(cur_edge)
            if not sorted_selected_edges:
                sorted_selected_edges.append((cur_edge, cur_edge_score))
            else:
                for index, (sorted_edge, sorted_score) in enumerate(sorted_selected_edges):
                    if cur_edge_score > sorted_score:
                        sorted_selected_edges.insert(index, (cur_edge, cur_edge_score))
                        break
                else:
                    sorted_selected_edges.append((cur_edge, cur_edge_score))
 
        for edge, _ in sorted_selected_edges:
            self._iter_prune(edge[0])

    def print_graph(self):
        for edge in self._meta_graph.edges():
            logger.info(f"Cur edge is {edge} ....")

    def _cache_pruned_biedges(self, deleted_edges):
        for edge in deleted_edges:
            reversed_edge = (edge[1], edge[0])
            if reversed_edge in self.bidirected_edges and reversed_edge not in self.searched_edges:
                self.pruned_edges.append(reversed_edge)

    def _delete_upstream_edges(self, front_end_metric):
        ''' search edge between front_end_metric and its neighbors
            add function: split all next pods to upstream node of fv node.
        '''
        selected_edges = []
        for edge in self._meta_graph.edges():
            if front_end_metric == edge[0]:
                selected_edges.append(edge)

        # seach next pods to upstream node
        up_nodes = set([edge[1] for edge in selected_edges])
        for edge in self._meta_graph.edges():
            for up_node in up_nodes:
                if edge not in selected_edges and edge[1] == up_node:
                    selected_edges.append(edge)

        self._meta_graph.remove_edges_from(selected_edges)

    def _delete_biedges(self):
        self._meta_graph.remove_edges_from(self.pruned_edges)

    @property
    def meta_graph(self):
        return self._meta_graph

    def run(self, front_end_metric, pruned_front=True):
        self._iter_prune(front_end_metric)
        self._delete_biedges()

        if pruned_front:
            self._delete_upstream_edges(front_end_metric)

        return self._meta_graph
