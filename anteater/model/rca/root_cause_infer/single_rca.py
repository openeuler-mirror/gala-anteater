# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：single_rca.py
Author: h00568282/huangbin 
Create Date: 2023/12/29 17:24
Notes:

"""
import re
from anteater.utils.log import logger



class SingleRca():
    def __init__(self, config, pod_id, node_id, anomaly_result):
        self.anomaly_result = anomaly_result
        self.pod_id = pod_id
        self.node_id = node_id
        self.cause_root = config.get("cause_root", [])
        self.pattern = r"gala_gopher_([a-zA-Z]+)_"

    def _get_result_template(self):
        return {
            'pod_id': self.pod_id,
            'node_id': self.node_id,
            'root_casue_metric': ''
        }

    def __call__(self):
        result_dict = self._get_result_template()

        anomaly_scores = {item.get("metric").split("@")[0]: item.get("score")
                          for item in self.anomaly_result.get("Resource").get("cause_metrics")}
        if not anomaly_scores:
            logger.info(f"node_id: {self.node_id} has no anomaly metric scores...")
            return result_dict

        sorted_score = sorted(anomaly_scores.items(), key=lambda x: -x[1])
        root_cause_metric = sorted_score[0][0]
        if self.cause_root:
            for key, _ in sorted_score:
                match = re.search(self.pattern, key)
                if match:
                    entity = match[1]
                    if entity in self.cause_root:
                        root_cause_metric = key
                        break

        result_dict['root_casue_metric'] = root_cause_metric

        return result_dict


def main():
    pass


if __name__ == "__main__":
    main()
