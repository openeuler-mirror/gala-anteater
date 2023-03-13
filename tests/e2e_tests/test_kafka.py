import json
from collections import defaultdict
from os import path, sep
from os.path import realpath
from typing import Any, Dict

from anteater.config import KafkaConf
from anteater.utils.log import logger


class TestKafkaProvider:
    def __init__(self, conf: KafkaConf) -> None:
        self.model_topic = conf.model_topic
        self.meta_topic = conf.meta_topic

        self.metadata = []
        self.updating()

        self.received_messages = defaultdict(list)

    def updating(self):
        folder_path = path.dirname(path.dirname(realpath(__file__)))
        file = path.join(folder_path, sep.join(["data", "metadata.json"]))

        with open(file) as f:
            data = json.load(f)
            self.metadata.extend(data)

    def get_metadata(self, entity_name):
        for item in self.metadata:
            if item.get("entity_name", "") == entity_name:
                return item.get("keys", {})
        logger.error(f"Unknown entity_name {entity_name} in metadata.json")
        return {}

    def send_message(self, message: Dict[str, Any]):
        logger.info(f"Sending the abnormal message to Kafka: {str(message)}")
        topic = self.model_topic
        message = json.dumps(message).encode('utf-8')
        self.received_messages[topic].append(message)
