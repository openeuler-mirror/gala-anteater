# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：path_builder.py
Author: h00568282/huangbin 
Create Date: 2023/12/29 15:58
Notes:

"""
import networkx as nx
from anteater.utils.log import logger



class MinPathSearcher():
    def __init__(self, args, time_stamp, meta_graph, machine_anomaly_scores):
        self.args = args
        self.time_stamp = time_stamp
        self.meta_graph = meta_graph
        self.undir_g = self._init_undir_graph()

        self.machine_anomaly_scores = machine_anomaly_scores
        self.output_dict = dict()

        self.error_count = 0
        # 根因节点列表
        self.has_hint_pods = []
        self.filter_kpis = ['gala_gopher_sli_rtt_nsec', 'gala_gopher_sli_tps']

    @staticmethod
    def _filter_kpis(location_result, filter_kpis=None):
        delete_results = []
        if not filter_kpis:
            return
        for kpi in filter_kpis:
            for result in location_result:
                root_cause_metric = result[0]
                if kpi in root_cause_metric:
                    delete_results.append(result)
                    break

        for result in delete_results:
            location_result.remove(result)

    def _find_dijkstra_path(self, root_cause_pod, front_end_pod):
        # 搜索两点间最短路径
        try:
            shortest_path = nx.dijkstra_path(self.undir_g, root_cause_pod, front_end_pod)
            shortest_path_len = nx.dijkstra_path_length(self.undir_g, root_cause_pod, front_end_pod)
            if shortest_path_len == 0:
                score = 1
            else:
                score = 1 / shortest_path_len
        except nx.exception.NetworkXNoPath as e:
            self.error_count += 1
            logger.info(f"nx.exception.NetworkXNoPath: {e}")
            shortest_path = None
            score = None

        return shortest_path, score

    def _init_undir_graph(self):
        undir_g = nx.Graph(self.meta_graph)
        for s, t in undir_g.edges():
            scores = self.meta_graph.nodes.get(s).get("anomaly_score") + self.meta_graph.nodes.get(t).get(
                "anomaly_score")
            if scores == 0:
                scores = 1e-6
            undir_g.edges[s, t].update(
                {"weight": (2 / scores)})

        return undir_g

    def _get_same_pods_index(self):
        ''' 找到topk结果中相同entity的索引 '''
        same_index = []
        unique_pods = []

        for index, pod in enumerate(self.has_hint_pods):
            if pod not in unique_pods:
                unique_pods.append(pod)
            else:
                same_index.append(index)

        return same_index

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

    def _get_neighbor_pods(self, cur_pod):
        up_nodes = []
        for pod in list(self.meta_graph[cur_pod]):
            if pod not in self.has_hint_pods:
                up_nodes.append(pod)

        down_nodes = []
        for node in self.meta_graph.nodes():
            if cur_pod in list(self.meta_graph[node]) and node not in self.has_hint_pods:
                down_nodes.append(node)

        up_nodes.extend(down_nodes)

        return up_nodes

    def _get_neighbor_root_cause_metric(self, valid_pod):
        metric_scores = self.machine_anomaly_scores[valid_pod]
        metric_scores = sorted(metric_scores.items(), key=lambda x: -x[1])
        try:
            raw_metric = metric_scores[0][0].split("*")[0]
        except Exception:
            raw_metric = ""

        return raw_metric

    def infer_causal_path(self, location_result, front_end_pod):
        self._filter_kpis(location_result, self.filter_kpis)
        ac_k = self.args.get("ac_k")  # topk results

        top_k = min(len(location_result), ac_k)
        for top_i in range(top_k):
            root_cause_metric = location_result[top_i][0]
            root_cause_pod = root_cause_metric.split("*")[1]
            self.has_hint_pods.append(root_cause_pod)
            logger.info(f"""timestamp: {self.time_stamp}, top{top_i + 1}, fv_pod:{front_end_pod},"""
                  f"""root_cause_metric:{root_cause_metric}, rw_score:{location_result[top_i][1]}""")

            shortest_path, score = self._find_dijkstra_path(root_cause_pod, front_end_pod)

            self.output_dict["top" + str(top_i + 1)] = {
                "root_cause": root_cause_metric,
                "root_cause_path": shortest_path,
                "score": score
            }

        # 补足到符合要求输出的数量
        # 补充的节点不加入has_hint_pods
        if top_k < ac_k:
            for add_index in range(top_k, ac_k):
                top_add_index = "top" + str(add_index + 1)
                self.output_dict[top_add_index] = self.output_dict["top" + str(top_k)]
                logger.info(f"""timestamp: {self.time_stamp}, top{add_index + 1}, fv_pod:{front_end_pod},"""
                      f"""root_cause_metric:{self.output_dict[top_add_index]["root_cause"]}, rw_score:0""")

    def add_neighbor_anomaly_pods(self, front_end_pod):
        # 增加最近邻搜索异常节点
        same_index_list = self._get_same_pods_index()
        diff_pods_len = len(same_index_list)

        if same_index_list:
            total_anomaly_scores = self._process_anomaly_scores(self.machine_anomaly_scores)
            searched_pods = []
            neighbor_pods = []
            for pod in self.has_hint_pods:
                if pod in searched_pods:
                    continue
                neighbor_pods.extend(self._get_neighbor_pods(pod))
                searched_pods.append(pod)

            sorted_neighbor_pods = []
            for cur_pod in neighbor_pods:
                cur_pod_score = total_anomaly_scores.get(cur_pod, 0)
                if not sorted_neighbor_pods:
                    sorted_neighbor_pods.append((cur_pod, cur_pod_score))
                else:
                    for index, (sorted_edge, sorted_score) in enumerate(sorted_neighbor_pods):
                        if cur_pod_score > sorted_score:
                            sorted_neighbor_pods.insert(index, (cur_pod, cur_pod_score))
                            break
                    else:
                        sorted_neighbor_pods.append((cur_pod, cur_pod_score))

            valid_len = min(len(sorted_neighbor_pods), diff_pods_len)
            logger.info(f"valid neighbor pods sum scores: {sorted_neighbor_pods}.")

            valid_neighbor_pods = sorted_neighbor_pods[:valid_len]  # (pod, total_score)

            if valid_neighbor_pods:
                cur_same_index = 0
                for valid_pod, _ in valid_neighbor_pods:
                    shortest_path, score = self._find_dijkstra_path(valid_pod, front_end_pod)

                    metric = self._get_neighbor_root_cause_metric(valid_pod)
                    if not metric:
                        continue
                    root_cause_metric = f"{metric}*{valid_pod}"
                    top_index = same_index_list[cur_same_index] + 1
                    self.output_dict["top" + str(top_index)] = {
                        "root_cause": root_cause_metric,
                        "root_cause_path": shortest_path,
                        "score": score
                    }
                    logger.info(
                        f"""timestamp: {self.time_stamp}, top{str(top_index)}, fv_pod:{front_end_pod}, """
                        f"""root_cause_metric:{root_cause_metric}, rw_score:0""")

                    cur_same_index += 1
                    if cur_same_index == diff_pods_len:
                        break

    def __call__(self, location_result, front_end_pod, *args, **kwargs):
        # 根据随机游走结果推理路径
        self.infer_causal_path(location_result, front_end_pod)
        # 去除重复pod, 补充最近邻pod作为候选
        self.add_neighbor_anomaly_pods(front_end_pod)

        return self.output_dict, self.error_count


def main():
    pass


if __name__ == "__main__":
    main()
