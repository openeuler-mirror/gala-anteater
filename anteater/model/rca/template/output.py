from collections import defaultdict
from dataclasses import dataclass

from anteater.model.rca.template.cause import Kpi, Metric, PathNode
from anteater.model.rca.template.kafka import KafkaProvider


@dataclass
class MetricInfo:
    metric_labels: defaultdict(str)
    metric_desc: defaultdict(str)
    metric_score: defaultdict(float)


def get_template(timestamp, event_id, abnormal_kpi, cause_metrics, keywords):
    cause_metrics_dict = []
    for c in cause_metrics:
        res = c.to_dict()
        # for res_path_item in res["path"]:
        #     del res_path_item['path'] 
        cause_metrics_dict.append(res)
    result = {
        'Timestamp': "1703669803936",
        'event_id': event_id,
        'Attributes': {
            'event_id': event_id,
            'event_source': 'nankai-root-cause-location'
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
    # result = json.dumps(result, ensure_ascii=False)

    return result


def load_keywords():
    entity_keywords = {
        "proc": "process", "thread": "process", "container": "process", "sli": "process",
        "block": "disk", "disk": "disk", "tcp_link": "network", "nic": "network", "qdisc": "network",
        "endpoint": "network", "cpu": "cpu", "mem": "memory", "fs": "file system"
    }
    return entity_keywords


def get_metric_info(metric_name, metric_labels, metric_desc, metric_score, machine_id):
    metric_info = None
    # try:
    metric_info = MetricInfo(metric_labels[machine_id + "@" + metric_name],
                             metric_desc[machine_id + "@" + metric_name], metric_score[machine_id + "@" + metric_name])
    # except KeyError:
    #     metric_name_new = machine_id + "@" + metric_name
    #     logger.logger.error(f"Unknown metric_name {metric_name_new} in MetricInfo")
    return metric_info


def get_cause_metrics(cause_outputs, metric_info, metric_to_entity, entity_keywords, machine_id, top_k, pod_states, pod_names, pod_instances, pod_jobs):
    metric_name = cause_outputs[top_k]["root_cause"]
    metric_labels = metric_info.metric_labels
    metric_desc = metric_info.metric_desc
    metric_score = metric_info.metric_score

    # metric_entity = metric_to_entity[machine_id][metric_name]
    metric_entity = ""

    metric_keyword = None
    for k in list(entity_keywords.keys()):
        # if k in metric_entity:
        if k in metric_name:
            metric_keyword = entity_keywords[k]
            break
    top_path = []
    if "root_cause_path" in cause_outputs[top_k] and cause_outputs[top_k]["root_cause_path"]:
        for path_metric in cause_outputs[top_k]["root_cause_path"]:
            top_path.append(PathNode(path_metric, pod_names.get(path_metric, ''), pod_instances.get(path_metric, ''), pod_jobs.get(path_metric, ''), pod_states.get(path_metric, 'normal')))

    cause_metric = Metric(metric_name, metric_entity, metric_labels, metric_desc,
                          metric_score, metric_keyword, path=top_path)
    return metric_keyword, cause_metric


def set_metric_info(metric_labels, metric_desc, metric_score, anomaly_outputs):
    machine_id = anomaly_outputs['Attributes']['event_id'].split("_")[1]

    metric_labels[machine_id + "@" + anomaly_outputs["Resource"]
                  ["metric"].split("@")[0]] = anomaly_outputs["Resource"]["labels"]
    metric_desc[machine_id + "@" + anomaly_outputs["Resource"]
                ["metric"].split("@")[0]] = anomaly_outputs["Resource"]["description"]
    metric_score[machine_id + "@" + anomaly_outputs["Resource"]
                 ["metric"].split("@")[0]] = anomaly_outputs["Resource"]["score"]

    for m in anomaly_outputs["Resource"]["cause_metrics"]:
        metric_labels[machine_id + "@" + m["metric"].split("@")[0]] = m["labels"]
        metric_desc[machine_id + "@" + m["metric"].split("@")[0]] = m["description"]
        metric_score[machine_id + "@" + m["metric"].split("@")[0]] = m["score"]
    return metric_labels, metric_desc, metric_score


def output(cause_outputs, anomaly_outputs, all_anomaly_results, pod_states, pod_names, pod_instances, pod_jobs):

    # logger.logger.info(f"cause_outputs: {cause_outputs}")

    entity_keywords = load_keywords()

    anomaly_timestamp = anomaly_outputs["Timestamp"]
    anomaly_event_id = anomaly_outputs['Attributes']['event_id']
    anomaly_kpi_name = anomaly_outputs["Resource"]["metric"].split("@")[0]
    anomaly_machine_id = anomaly_outputs['Attributes']['event_id'].split("_")[1]

    metric_labels = {}
    metric_desc = {}
    metric_score = {}

    metric_labels, metric_desc, metric_score = set_metric_info(
        metric_labels, metric_desc, metric_score, anomaly_outputs)
    
    for anomaly_result in all_anomaly_results:
        timestamp = anomaly_result["Timestamp"]
        event_id = anomaly_result['Attributes']['event_id']
        kpi_name = anomaly_result["Resource"]["metric"].split("@")[0]
        machine_id = anomaly_result['Attributes']['event_id'].split("_")[1]
        metric_labels, metric_desc, metric_score = set_metric_info(
        metric_labels, metric_desc, metric_score, anomaly_result)

    kpi_labels = metric_labels.get(anomaly_machine_id + "@" + anomaly_kpi_name, 'Not exist')
    kpi_desc = metric_desc.get(anomaly_machine_id + "@" + anomaly_kpi_name, 'Not exist')
    kpi_score = metric_score.get(anomaly_machine_id + "@" + anomaly_kpi_name, 'Not exist')
    abnormal_kpi = Kpi(anomaly_kpi_name, "", kpi_labels, kpi_desc, kpi_score)

    keywords = []
    cause_metrics_param = []

    if "top1" in cause_outputs:
        metric1_name = cause_outputs["top1"]["root_cause"]

        metric_info1 = get_metric_info(cause_outputs["top1"]["root_cause"].split("*")[0],
                                       metric_labels, metric_desc, metric_score, cause_outputs["top1"]["root_cause"].split("*")[1])
        metric_keyword, cause_metric = get_cause_metrics(
            cause_outputs, metric_info1, {}, entity_keywords, cause_outputs["top1"]["root_cause"].split("*")[1], "top1", pod_states, pod_names, pod_instances, pod_jobs)
        keywords.append(metric_keyword)
        cause_metrics_param.append(cause_metric)

    if "top2" in cause_outputs:
        metric2_name = cause_outputs["top2"]["root_cause"]

        metric_info2 = get_metric_info(cause_outputs["top2"]["root_cause"].split("*")[0],
                                       metric_labels, metric_desc, metric_score, cause_outputs["top2"]["root_cause"].split("*")[1])
        metric_keyword, cause_metric = get_cause_metrics(
            cause_outputs, metric_info2, {}, entity_keywords, cause_outputs["top2"]["root_cause"].split("*")[1], "top2", pod_states, pod_names, pod_instances, pod_jobs)
        if metric_keyword not in keywords:
            keywords.append(metric_keyword)
        cause_metrics_param.append(cause_metric)

    if "top3" in cause_outputs:
        metric3_name = cause_outputs["top3"]["root_cause"]

        metric_info3 = get_metric_info(cause_outputs["top3"]["root_cause"].split("*")[0],
                                       metric_labels, metric_desc, metric_score, cause_outputs["top3"]["root_cause"].split("*")[1])
        metric_keyword, cause_metric = get_cause_metrics(
            cause_outputs, metric_info3, {}, entity_keywords, cause_outputs["top3"]["root_cause"].split("*")[1], "top3", pod_states, pod_names, pod_instances, pod_jobs)
        if metric_keyword not in keywords:
            keywords.append(metric_keyword)
        cause_metrics_param.append(cause_metric)

    res = get_template(anomaly_timestamp, anomaly_event_id, abnormal_kpi, cause_metrics_param, keywords)
    kafka_provider = KafkaProvider()
    kafka_provider.send_message(res)
