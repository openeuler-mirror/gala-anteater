import re

from anteater.template.template import Template


class SysAnomalyTemplate(Template):
    """The sys anomaly report template"""
    def __init__(self, timestamp, machine_id, metric_id, entity_name):
        super().__init__(timestamp, machine_id, metric_id, entity_name)

    def get_template(self):
        timestamp = round(self.timestamp.timestamp() * 1000)

        result = {
            "Timestamp": timestamp,
            "event_id": f"{timestamp}_{self.entity_id}",
            "Attributes": {
                "entity_id": self.entity_id,
                "event_id": f"{timestamp}_{self.entity_id}",
                "event_type": "app"
            },
            "Resource": {
                "metric_id": self.metric_id,
                "labels": self.labels,
                "cause_metrics": self.cause_metrics,
                "description": self.description
            },
            "SeverityText": "WARN",
            "SeverityNumber": 13,
            "Body": f"{self.timestamp.strftime('%c')} WARN, SYS may be impacting performance issues."
        }

        return result
