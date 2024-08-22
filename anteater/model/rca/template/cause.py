


class Kpi:

    def __init__(self, metric_id, entity_id, metric_labels, desc, score):
        self.metric_id = metric_id
        self.entity_id = entity_id
        self.metric_labels = metric_labels
        self.desc = desc
        self.score = score

    def to_dict(self):
        kpi_dict = {}
        kpi_dict["metric_id"] = self.metric_id
        kpi_dict["entity_id"] = self.entity_id
        kpi_dict["metric_labels"] = self.metric_labels
        kpi_dict["desc"] = self.desc
        kpi_dict["score"] = self.score
        return kpi_dict


class Metric:

    def __init__(self, metric_id, entity_id, metric_labels, desc, score, keyword, path=None):
        if "*" in metric_id:
            metric_id = metric_id.replace('*','@')
        self.metric_id = metric_id
        self.entity_id = entity_id
        self.metric_labels = metric_labels
        self.desc = desc
        self.score = score
        self.keyword = keyword
        self.path = path

    def to_dict(self):
        metric_dict = {}
        metric_dict["metric_id"] = self.metric_id
        metric_dict["entity_id"] = self.entity_id
        metric_dict["metric_labels"] = self.metric_labels
        metric_dict["desc"] = self.desc
        metric_dict["keyword"] = self.keyword
        metric_dict["score"] = self.score
        metric_dict["path"] = []
        if self.path is not None:
            for p in self.path:
                metric_dict["path"].append(p.to_dict())
        return metric_dict

class PathNode:

    def __init__(self, pod_id, pod, instance, job, pod_state):
        self.pod_id = pod_id
        self.pod = pod
        self.instance = instance
        self.job = job
        self.pod_state = pod_state

    def to_dict(self):
        path_node_dict = {}
        path_node_dict["pod_id"] = self.pod_id
        path_node_dict["pod"] = self.pod
        path_node_dict["instance"] = self.instance
        path_node_dict["job"] = self.job
        path_node_dict["pod_state"] = self.pod_state
        return path_node_dict


class Cause:
    def __init__(self, metric_id, entity_id, cause_score, path):
        self.metric_id = metric_id
        self.entity_id = entity_id
        self.cause_score = cause_score
        self.path = path

    def to_dict(self):
        res = {
            'metric_id': self.metric_id,
            'entity_id': self.entity_id,
            'cause_score': self.cause_score
        }
        return res
