# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：prob_matrix_builder.py
Author: h00568282/huangbin 
Create Date: 2023/12/29 15:58
Notes:

"""
import numpy as np
import pandas as pd


class ProbMatrixBuilder():
    def __init__(self, args):
        self.args = args
        pass

    def __call__(self, graph, v_fe, *args, **kwargs):
        transfer_mat = self.build_probablity_matrix(graph, v_fe)

        return transfer_mat

    @staticmethod
    def remove_self(list_a, a):
        list_a = list(list_a)
        if a in list_a:
            list_a.remove(a)
            return list_a
        else:
            return list_a

    def generate_df(self, graph):
        """
        将各节点的指标值整合成DataFrame
        """
        dict_tmp = {}
        for i in graph.nodes():
            dict_tmp[i] = graph.nodes.get(i).get("timelist")
        return pd.DataFrame(dict_tmp)

    def partial_corr_c(self, graph, v_j, v_fe, df, corr_type):
        """
        计算下一个节点v_j和前端节点v_fe的偏相关
        """
        if v_j == v_fe:
            return 1.

        parent_v_j = self.remove_self(graph.pred.get(v_j), v_j)
        parent_v_fe = self.remove_self(graph.pred.get(v_fe), v_fe)

        confounder = list(set(parent_v_j) | set(parent_v_fe))
        if v_fe in confounder:
            confounder.remove(v_fe)
        if v_j in confounder:
            confounder.remove(v_j)

        if len(df[v_fe].unique()) == 1:
            return 0.
        if len(df[v_j].unique()) == 1:
            return 0.
        for item in confounder:
            if len(df[item].unique()) == 1:
                confounder.remove(item)

        import pingouin as pg
        return abs(pg.partial_corr(data=df, x=v_fe, y=v_j, covar=confounder, method=corr_type)['r'].values[0])

    def transfer_prob(self, graph, nodes: tuple, df, corr_type, corr_prop):
        """
        综合考虑异常分数和偏相关计算下一个节点v_j的跳转概率
        """
        v_i, v_j, v_fe = nodes[0], nodes[1], nodes[2]
        v_i_score = graph.nodes.get(v_i).get("anomaly_score")
        v_j_score = graph.nodes.get(v_j).get("anomaly_score")
        transfer_anomaly = v_j_score / (v_i_score + v_j_score) if (v_i_score + v_j_score) != 0 else 0
        transfer_corr = self.partial_corr_c(graph, v_j, v_fe, df, corr_type)
        return transfer_anomaly * (1 - corr_prop) + transfer_corr * corr_prop

    def build_probablity_matrix(self, graph, v_fe):
        """
        计算概率矩阵
        """
        r = self.args.get("r")
        remove_kpi = bool(self.args.get("remove_kpi"))
        corr_type = self.args.get("corr_type")
        corr_prop = self.args.get("corr_prop")

        n = len(graph.nodes())
        df = self.generate_df(graph)
        transfer_mat = pd.DataFrame(np.zeros((n, n), dtype=np.float64),
                                    index=graph.nodes(), columns=graph.nodes())
        for i in graph.nodes():
            successor = self.remove_self(graph.succ.get(i), i)

            for j in successor:
                transfer_mat[i][j] = r * self.transfer_prob(graph, (i, j, v_fe), df, corr_type, corr_prop)

            if remove_kpi:
                parents_i = self.remove_self(self.remove_self(graph.pred.get(i), i), v_fe)
            else:
                parents_i = self.remove_self(graph.pred.get(i), i)
            for j in parents_i:
                transfer_mat[i][j] = self.transfer_prob(graph, (i, j, v_fe), df, corr_type, corr_prop)

            c_self = self.transfer_prob(graph, (i, i, v_fe), df, corr_type, corr_prop)
            if c_self > transfer_mat[i].max():
                transfer_mat[i][i] = c_self - transfer_mat[i].max()

            s = transfer_mat[i].sum()
            if s == 0:
                continue
            for j in set(parents_i + successor):
                transfer_mat[i][j] = transfer_mat[i][j] / s
            transfer_mat[i][i] = transfer_mat[i][i] / s

        return transfer_mat


def main():
    pass


if __name__ == "__main__":
    main()
