# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：utils.py
Author: h00568282/huangbin 
Create Date: 2023/12/27 10:23
Notes:

"""

import networkx as nx


def create_meta_graph(meta_data: dict) -> nx.DiGraph:
    """
    构建元图
    """
    meta_graph = nx.DiGraph()
    # 添加节点
    for node in meta_data:
        meta_graph.add_node(node)
    # 添加有向边
    for node, neighbors in meta_data.items():
        for neighbor in neighbors:
            meta_graph.add_edge(node, neighbor)
    return meta_graph


def main():
    pass


if __name__ == "__main__":
    main()
