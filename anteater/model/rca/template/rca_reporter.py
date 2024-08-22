import os
import json
from collections import defaultdict
from dataclasses import dataclass

from anteater.utils.log import logger
from anteater.model.rca.template.cause import Kpi, Metric, PathNode
from anteater.model.rca.template.kafka import KafkaProvider


@dataclass
class MetricInfo:
    metric_labels: defaultdict(str)
    metric_desc: defaultdict(str)
    metric_score: defaultdict(float)


class RcaReporter():
    def __init__(self, cause_outputs, anomaly_outputs, all_anomaly_results, pod_states, pod_names, pod_instances,
                 pod_jobs):
        self.cause_outputs = cause_outputs
        self.anomaly_outputs = anomaly_outputs
        self.all_anomaly_results = all_anomaly_results
        self.pod_states = pod_states
        self.pod_names = pod_names
        self.pod_instances = pod_instances
        self.pod_jobs = pod_jobs

        self.entity_keywords = self.load_keywords()

    def load_keywords(self):
        entity_keywords = {
            "proc": "process", "thread": "process", "container": "process", "sli": "process",
            "block": "disk", "disk": "disk", "tcp_link": "network", "nic": "network", "qdisc": "network",
            "endpoint": "network", "cpu": "cpu", "mem": "memory", "fs": "file system"
        }
        return entity_keywords

    @staticmethod
    def get_metric_info(metric_name, metric_labels, metric_desc, metric_score, machine_id):
        metric_info = None
        try:
            metric_info = MetricInfo(metric_labels[machine_id + "@" + metric_name],
                                     metric_desc[machine_id + "@" + metric_name],
                                     metric_score[machine_id + "@" + metric_name])
        except KeyError:
            metric_name_new = machine_id + "@" + metric_name
            logger.logger.error(f"Unknown metric_name {metric_name_new} in MetricInfo")

        return metric_info

    def get_cause_metrics(self, metric_info, top_k):
        metric_name = self.cause_outputs[top_k]["root_cause"]
        metric_labels = metric_info.metric_labels
        metric_desc = metric_info.metric_desc
        metric_score = metric_info.metric_score

        # metric_entity = metric_to_entity[machine_id][metric_name]
        metric_entity = ""

        metric_keyword = None
        for k in list(self.entity_keywords.keys()):
            # if k in metric_entity:
            if k in metric_name:
                metric_keyword = self.entity_keywords[k]
                break
        top_path = []
        if "root_cause_path" in self.cause_outputs[top_k] and self.cause_outputs[top_k]["root_cause_path"]:
            for path_metric in self.cause_outputs[top_k]["root_cause_path"]:
                top_path.append(
                    PathNode(path_metric, self.pod_names.get(path_metric, ''), self.pod_instances.get(path_metric, ''),
                             self.pod_jobs.get(path_metric, ''), self.pod_states.get(path_metric, 'normal')))

        cause_metric = Metric(metric_name, metric_entity, metric_labels, metric_desc,
                              metric_score, metric_keyword, path=top_path)

        return metric_keyword, cause_metric

    @staticmethod
    def set_metric_info(metric_labels, metric_desc, metric_score, anomaly_outputs):
        machine_id = anomaly_outputs['Attributes']['event_id'].split("_")[1]

        metric_labels[machine_id + "@" + anomaly_outputs["Resource"]
        ["metric"].split("@")[0]] = anomaly_outputs["Resource"]["labels"]
        metric_desc[machine_id + "@" + anomaly_outputs["Resource"]
        ["metric"].split("@")[0]] = anomaly_outputs["Resource"]["description"]
        metric_score[machine_id + "@" + anomaly_outputs["Resource"]
        ["metric"].split("@")[0]] = anomaly_outputs["Resource"]["score"]

        for m in anomaly_outputs["Resource"]["root_causes"]:
            metric_labels[machine_id + "@" + m["metric"].split("@")[0]] = m["labels"]
            metric_desc[machine_id + "@" + m["metric"].split("@")[0]] = m.get("description", "")
            metric_score[machine_id + "@" + m["metric"].split("@")[0]] = m["score"]

        return metric_labels, metric_desc, metric_score

    def __call__(self, kpis, all_fv_metrics, *args, **kwargs):
        anomaly_timestamp = self.anomaly_outputs["Timestamp"]
        anomaly_event_id = self.anomaly_outputs['Attributes']['event_id']
        anomaly_kpi_name = self.anomaly_outputs["Resource"]["metric"].split("@")[0]
        anomaly_machine_id = self.anomaly_outputs['Attributes']['event_id'].split("_")[1]

        metric_labels = {}
        metric_desc = {}
        metric_score = {}

        metric_labels, metric_desc, metric_score = self.set_metric_info(
            metric_labels, metric_desc, metric_score, self.anomaly_outputs)

        for anomaly_result in self.all_anomaly_results.values():
            metric_labels, metric_desc, metric_score = self.set_metric_info(
                metric_labels, metric_desc, metric_score, anomaly_result)

        kpi_labels = metric_labels.get(anomaly_machine_id + "@" + anomaly_kpi_name, 'Not exist')
        kpi_desc = metric_desc.get(anomaly_machine_id + "@" + anomaly_kpi_name, 'Not exist')
        kpi_score = metric_score.get(anomaly_machine_id + "@" + anomaly_kpi_name, 'Not exist')
        abnormal_kpi = Kpi(anomaly_kpi_name, "", kpi_labels, kpi_desc, kpi_score)

        keywords = []
        cause_metrics_param = []
        for i in range(3):
            top_k = f"top{str(i + 1)}"
            if top_k in self.cause_outputs:
                r_metric, r_machine = self.cause_outputs[top_k]["root_cause"].split("*")
                metric_info = self.get_metric_info(r_metric, metric_labels, metric_desc, metric_score, r_machine)
                metric_keyword, cause_metric = self.get_cause_metrics(metric_info, top_k)

                if metric_keyword not in keywords:
                    keywords.append(metric_keyword)
                cause_metrics_param.append(cause_metric)

        res = self.get_template(anomaly_timestamp, anomaly_event_id, abnormal_kpi, cause_metrics_param, keywords, kpis,
                                all_fv_metrics)

        return res

    @staticmethod
    def get_template(timestamp, event_id, abnormal_kpi, cause_metrics, keywords, kpis, all_fv_metrics):
        cause_metrics_dict = []
        for c in cause_metrics:
            res = c.to_dict()
            cause_metrics_dict.append(res)

        result = {
            'Timestamp': timestamp,
            'event_id': event_id,
            'Attributes': {
                'event_id': event_id,
                'event_source': 'root-cause-inference'
            },
            'Resource': {
                'abnormal_kpi': abnormal_kpi.to_dict(),
                'cause_metrics': cause_metrics_dict
            },
        }
        result['desc'] = abnormal_kpi.desc
        if len(cause_metrics) > 0:
            result['top1'] = cause_metrics[0].metric_id + "异常"
        if len(cause_metrics) > 1:
            result['top2'] = cause_metrics[1].metric_id + "异常"
        if len(cause_metrics) > 2:
            result['top3'] = cause_metrics[2].metric_id + "异常"
        result['keywords'] = keywords
        result['SeverityText'] = 'WARN'
        result['SeverityNumber'] = 13
        result['Body'] = 'A cause inferring event for an abnormal event'
        result["kpis"] = kpis
        result["all_fv_metrics"] = all_fv_metrics

        return result
