from abc import abstractmethod


class Template:
    """The app anomaly template"""
    def __init__(self, timestamp, machine_id, metric_id, entity_name):
        self.timestamp = timestamp
        self.machine_id = machine_id
        self.metric_id = metric_id
        self.entity_name = entity_name
        self.labels = {}

        self.entity_id = ""
        self.description = ""
        self.cause_metrics = {}

    @abstractmethod
    def get_template(self):
        pass
