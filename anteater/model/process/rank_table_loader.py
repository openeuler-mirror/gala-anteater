# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：rank_table_loader.py
Author: h00568282/huangbin 
Create Date: 2024/10/24 17:07
Notes:

"""
import json

NULL_RANK_TABLE_VALUE = 'null'
PP_KEY = 'pp'


class CommunicationDomain:
    def __init__(self, json_data):
        self.pp = json_data.get("pp", 1)
        self.tp = json_data.get("tp", 1)
        self.dp = json_data.get("dp", 1)


class RankTableLoader():
    def __init__(self, hccl_domain: dict, rank_table: dict, machine_ids_to_devices: dict):
        self.hccl_domain = CommunicationDomain(hccl_domain)
        self.rank_table = self.parse_rank_table(rank_table)
        self.nodes_ip, self.ranks, self.host_ids = self._init_nodes_ranks(machine_ids_to_devices)

    def _init_nodes_ranks(self, machine_ids_to_devices):
        nodes_ip, ranks = self.get_nodes_ranks()
        if not nodes_ip:
            nodes_ip = []
            ranks = []

            for node_ip, device_id_list in machine_ids_to_devices.items():
                ranks_num = len(ranks)
                flag = False
                for index, device_id in enumerate(device_id_list):
                    if device_id.isdigit():
                        flag = True
                        device_id_list[index] = int(device_id) + ranks_num
                if flag:
                    ranks.append(device_id_list)
                    nodes_ip.append(node_ip)

        host_ids = []
        for node_ip, device_id_list in machine_ids_to_devices.items():
            if device_id_list == [""]:
                host_ids.append(node_ip)

        return nodes_ip, ranks, host_ids

    def get_all_ranks(self):
        ''' 获取当前任务所有卡的rank_id '''
        all_ranks = []

        for node_rank in self.ranks:
            all_ranks.extend(node_rank)

        all_ranks = sorted(all_ranks)

        return all_ranks

    @staticmethod
    def parse_rank_table(rank_table_info):
        valid_info = "{}"
        for value in rank_table_info.values():
            if value == NULL_RANK_TABLE_VALUE:
                continue
            valid_info = value.replace("\'", "\"")

        return json.loads(valid_info)

    def get_nodes_ranks(self):
        ''' 获取所有节点和ranks
            @return:
                nodes_ip: list(str) = [ip1, ip2, ...]
                ranks: list(int) = [[0,1,2,3,4,5,6,7], [8,9,...], ...]
        '''
        nodes_ip = []
        ranks = []
        nodes_num = self.rank_table.get("serverCount", 0)
        if nodes_num:
            server_list = self.rank_table.get("serverList", [])
            for node_info in server_list:
                node_ip = node_info["serverId"]
                devices_info = node_info["device"]
                rank_ids = [int(device_info["rankId"]) for device_info in devices_info]
                nodes_ip.append(node_ip)
                ranks.append(rank_ids)

        return nodes_ip, ranks

    def get_group_nodes_by_ranks(self, group_ranks):
        group_node_ranks = {}

        for rank_id in group_ranks:
            for node_index, node_ranks in enumerate(self.ranks):
                if rank_id in node_ranks:
                    selected_node_ip = self.nodes_ip[node_index]
                    if selected_node_ip not in group_node_ranks.keys():
                        group_node_ranks[selected_node_ip] = [rank_id]
                    else:
                        group_node_ranks[selected_node_ip].append(rank_id)
                    break

        return group_node_ranks

    def get_ranks_by_nodes(self, nodes_ips):
        ranks = []
        for node_ip in nodes_ips:
            ranks.append(self._get_ranks_by_node(node_ip))

        return ranks

    def _get_ranks_by_node(self, node_ip):
        ranks = []
        try:
            index = self.nodes_ip.index(node_ip)
            return self.ranks[index]
        except ValueError:
            return ranks

    def get_group_nodes(self):
        '''
            根据rank_table message, pp分组参数返回pp组对应的ip
            @return:
            {
                0: ["192.168.10.101", "192.168.10.108", "192.168.10.102"]
                1: ["192.168.21.101", "192.168.11.108", "192.168.22.102"]
                ...
            }
        '''
        pp_num = self.hccl_domain.pp
        groups_nodes = {}
        all_ranks = []
        for node_rank in self.ranks:
            all_ranks.extend(node_rank)

        all_ranks = sorted(all_ranks)
        all_ranks_num = len(all_ranks)
        if all_ranks_num % pp_num != 0:
            return groups_nodes
        else:
            each_group_ranks_num = int(all_ranks_num / pp_num)
            for group_id in range(pp_num):
                selected_nodes_ips = []
                group_ranks = all_ranks[each_group_ranks_num * group_id: each_group_ranks_num * (group_id + 1)]
                for rank_id in group_ranks:
                    for node_index, node_ranks in enumerate(self.ranks):
                        if rank_id in node_ranks:
                            selected_node_ip = self.nodes_ip[node_index]
                            if selected_node_ip not in selected_nodes_ips:
                                selected_nodes_ips.append(selected_node_ip)
                            break

                groups_nodes[group_id] = selected_nodes_ips

        return groups_nodes


class GroupDataLoader:
    def __init__(self, hccl_domain: dict, rank_table: dict, machine_ids_to_devices: dict) -> list:
        self.rank_table_loader = RankTableLoader(hccl_domain, rank_table, machine_ids_to_devices)

    def get_group_ranks(self):
        ''' 根据group_info中的pp并行数，对rankid进行划分
            @params:
                group_info: dict, {pp:16, dp:4, tp:8 ...}
            @return:
                group_ranks: [
                    {0:[0,1,...,31]},
                    {1:[32,...,63]}
                    ...
                ]
        '''
        group_ranks = []
        all_rank_ids = self.rank_table_loader.get_all_ranks()
        rank_ids_num = len(all_rank_ids)
        pp_num = self.rank_table_loader.hccl_domain.pp

        if rank_ids_num % pp_num != 0:
            pp_num = 1
        each_group_ranks_num = rank_ids_num // pp_num
        for group_id in range(pp_num):
            group_ranks.append(
                all_rank_ids[each_group_ranks_num * group_id: each_group_ranks_num * (group_id + 1)]
            )

        return group_ranks