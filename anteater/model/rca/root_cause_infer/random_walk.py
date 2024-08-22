# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：random_walk.py
Author: h00568282/huangbin
Create Date: 2023/12/29 15:59
Notes:

"""


class RandomWalker():
    def __init__(self, args, random_walk_prob):
        self.num_loop = args.get("num_loop")
        self.beta = args.get("beta")
        self.random_walk_prob = random_walk_prob
        pass

    def __call__(self, graph, transfer_matrix, front_end_metric, *args, **kwargs):
        if len(graph.nodes()) == 0:
            rw_result = [
                (front_end_metric, self.num_loop)
            ]
            ext_rw = [(front_end_metric, 0)] * 10
            rw_result.extend(ext_rw)
        else:
            if self.beta == 1:
                rw_result = self.random_walk(graph, transfer_matrix, front_end_metric)
            else:
                rw_result = self.random_walk_second_order(graph, transfer_matrix, front_end_metric)

        location_result = [(k, v / self.num_loop) for k, v in rw_result]

        return rw_result, location_result

    def random_pick(self, some_list, probabilities, prob_thr):
        """
        根据跳转概率probabilities从some_list选择下一个跳转的节点
        """
        # x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if prob_thr < cumulative_probability:
                break

        return item

    def random_walk(self, graph, transfer_mat, v_fe):
        """
        一阶随机游走
        """

        # 初始化
        v_cur = v_fe
        visit_count = dict.fromkeys(graph.nodes(), 0)

        # 循环随机游走
        for idx in range(self.num_loop):
            v_cur = self.random_pick(transfer_mat.index.tolist(), transfer_mat[v_cur].values,
                                     self.random_walk_prob[idx])
            visit_count[v_cur] += 1

        visit_order = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)
        return visit_order

    def random_walk_second_order(self, graph, transfer_mat, v_fe):
        """
        二阶随机游走
        """
        # 初始化
        v_cur = v_fe
        v_pre = v_fe
        visit_count = dict.fromkeys(graph.nodes(), 0)
        danamic_transfer = dict.fromkeys(graph.nodes(), 0)

        # 循环随机游走
        for idx in range(self.num_loop):
            p_to_c = transfer_mat[v_pre][v_cur]
            sum = 0
            for key, value in transfer_mat[v_cur].iteritems():
                if value > 0:
                    danamic_transfer[key] = (1 - self.beta) * p_to_c + self.beta * transfer_mat[v_cur][key]
                    sum = sum + (1 - self.beta) * p_to_c + self.beta * transfer_mat[v_cur][key]
            if sum == 0:
                for key, value in danamic_transfer.items():
                    danamic_transfer[key] = 1 / len(danamic_transfer.items())
            else:
                for key, value in danamic_transfer.items():
                    danamic_transfer[key] = value / sum

            v_next = self.random_pick(danamic_transfer.keys(), danamic_transfer.values(), self.random_walk_prob[idx])
            visit_count[v_next] += 1

            v_pre = v_cur
            v_cur = v_next
        visit_order = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)

        return visit_order
