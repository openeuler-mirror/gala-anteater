# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：rca_metric_loader.py
Author: h00568282/huangbin 
Create Date: 2024/1/2 15:37
Notes:

"""
import copy
import os
import json
import warnings
import pandas as pd
from anteater.model.rca.common.utils import date_to_timestamp, smooth_data, _extrac_front_end_metric, filter_metric_not_candidates
from anteater.utils.log import logger

warnings.filterwarnings("ignore")

class RcaMetricLoader():
    def __init__(self, args, metric_candidates, special_sli_metrics):
        self.args = args
        self.metric_candidates = metric_candidates
        self.data_dir = args.get("data_dir")
        self.smooth_window = args.get("smooth_window")
        self.forward_extended_time = args.get("forward_extended_time")
        self.backward_extended_time = args.get("backward_extended_time")
        self.special_sli_metrics = special_sli_metrics

    def load_ano_results_from_json(self, machine_id):
        with open(os.path.join(self.data_dir, machine_id, "result.json")) as f:
            results = json.load(f)

        ano_result = results[0]
        logger.info(
            f"""当前故障case的开始时间为: {ano_result.get("Timestamp")}, """
            f"""utc时间字符串为: {pd.to_datetime(ano_result.get("Timestamp") // 1000, unit="s")}""")
        return ano_result

    def load_ano_results_from_kafka(self, machine_id):
        pass

    @staticmethod
    def extract_ano_scores(_machine_id, ano_result):
        metric_scores = {}
        if ano_result.get("Resource").get("root_causes"):
            metric_scores = {item.get("metric").split("@")[0] + "*" + _machine_id: float(item.get("score"))
                             for item in ano_result.get("Resource").get("root_causes")}
            metric_scores[ano_result.get("Resource").get("metric").split("@")[0] + "*" + _machine_id] = float(ano_result.get(
                "Resource").get("score"))

        return metric_scores

    def parse_sli_metric(self, _machine_id, ano_result, all_front_end_metrics):
        resource_metric = ano_result["Resource"]["metric"]
        if "@" in resource_metric:
            sli_front_end_metric = resource_metric.split('@')[0]
        else:
            sli_front_end_metric = resource_metric

        # 记录所有machine的front_end_metric
        if _machine_id not in all_front_end_metrics.keys():
            all_front_end_metrics[_machine_id] = [sli_front_end_metric]
        else:
            if sli_front_end_metric not in all_front_end_metrics[_machine_id]:
                all_front_end_metrics[_machine_id].append(sli_front_end_metric)

    def load_metric_ts(self, machine_id):
        # 读数据
        metric_ts = pd.read_csv(os.path.join(self.data_dir, machine_id, "metric.csv"))

        return metric_ts

    def preprocess_metric_ts(self, metric_ts, start_time, end_time):
        # 将date转为时间戳
        metric_ts["timestamp"] = metric_ts["timestamp"].apply(date_to_timestamp)
        # 使用候选指标名单过滤
        metric_ts = filter_metric_not_candidates(self.metric_candidates, metric_ts)
        # 数据平滑和缺失值补充
        metric_ts = smooth_data(df=metric_ts, window=self.smooth_window)
        now_metric = metric_ts[(metric_ts["timestamp"] >= start_time)
                               & (metric_ts["timestamp"] <= end_time)]
        del now_metric["timestamp"]

        # 去除时间窗口内保持不变的指标
        columns_to_delete = []
        for col in now_metric.columns:
            if len(now_metric[col].unique()) == 1 and col not in self.special_sli_metrics:
                columns_to_delete.append(col)

        now_metric = now_metric.drop(columns=columns_to_delete)

        return now_metric

    def load_data(self, machine_ids, all_machines_df, all_anomaly_results):
        ''' load time series and ano score '''
        all_metric_scores = {}
        machine_anomaly_scores = {}
        all_metrics_ts = {}
        all_front_end_metrics = {}
        # 加载异常检测结果
        for _machine_id in machine_ids:
            ano_result = copy.deepcopy(all_anomaly_results[_machine_id])
            # ano_result = self.load_ano_results_from_json(_machine_id)
            start_time = ano_result.get("Timestamp") // 1000 - self.forward_extended_time
            # end_time = result.get("Timestamp") // 1000 + backward_extended_time
            end_time = ano_result.get("Timestamp") // 1000
            # 收集machine的sli指标
            self.parse_sli_metric(_machine_id, ano_result, all_front_end_metrics)
            # 指标序列
            metric_ts = copy.deepcopy(all_machines_df[_machine_id])
            # metric_ts = self.load_metric_ts(_machine_id)
            now_metric_ts = self.preprocess_metric_ts(metric_ts, start_time, end_time)
            for col in now_metric_ts.columns:
                all_metrics_ts[col + "*" + _machine_id] = now_metric_ts[col]

            # 指标异常分数
            metric_scores = self.extract_ano_scores(_machine_id, ano_result)
            machine_anomaly_scores[_machine_id] = metric_scores
            all_metric_scores.update(metric_scores)

        all_metrics_ts = pd.DataFrame(all_metrics_ts)
        all_metrics_ts = all_metrics_ts.fillna(method="ffill").fillna(method="bfill").fillna(0)
        return all_metrics_ts, all_metric_scores, machine_anomaly_scores, all_front_end_metrics

def main():
    pass


if __name__ == "__main__":
    main()
