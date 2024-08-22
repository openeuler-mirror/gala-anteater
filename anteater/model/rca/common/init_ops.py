# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：init_ops.py
Author: h00568282/huangbin 
Create Date: 2024/1/2 21:20
Notes:

"""
import json
import os

from kafka.errors import KafkaTimeoutError
from kafka import KafkaProducer, KafkaConsumer

from anteater.model.rca.common.constants import INFER_CONFIG_PATH, EXT_OBSV_META_PATH, RULE_META_PATH, CAUSE_KEYWORD_PATH
from anteater.utils.log import logger

from anteater.model.rca.cause_inference.cause_keyword import cause_keyword_mgt
from anteater.model.rca.cause_inference.abnormal_event import AbnEvtMgt
from anteater.model.rca.cause_inference.rule_parser import rule_engine
from anteater.model.rca.cause_inference.config import init_infer_config, infer_config

def init_config():
    conf_path = os.environ.get('INFER_CONFIG_PATH') or INFER_CONFIG_PATH
    if not init_infer_config(conf_path):
        return False
    logger.init_logger('gala-inference', infer_config.log_conf)

    logger.logger.info('Load observe metadata success.')

    if not rule_engine.load_rule_meta_from_yaml(RULE_META_PATH):
        logger.logger.error('Load rule meta failed.')
        return False
    logger.logger.info('Load rule meta success.')

    if not cause_keyword_mgt.load_keywords_from_yaml(CAUSE_KEYWORD_PATH):
        logger.logger.error('Load cause keyword failed.')
        return False
    logger.logger.info('Load cause keyword success.')

    return True


def init_metadata_consumer():
    kafka_server = infer_config.kafka_conf.get('server')
    metadata_topic = infer_config.kafka_conf.get('metadata_topic')
    # metadata_consumer = KafkaConsumer(
    #     metadata_topic.get('topic_id'),
    #     bootstrap_servers=[kafka_server],
    #     group_id=metadata_topic.get('group_id')
    # )
    metadata_consumer = KafkaConsumer(
        metadata_topic.get('topic_id'),
        bootstrap_servers=[kafka_server],
    )
    return metadata_consumer


def init_kpi_consumer():
    kafka_server = infer_config.kafka_conf.get('server')
    kpi_kafka_conf = infer_config.kafka_conf.get('abnormal_kpi_topic')
    # kpi_consumer = KafkaConsumer(
    #     kpi_kafka_conf.get('topic_id'),
    #     bootstrap_servers=[kafka_server],
    #     group_id=kpi_kafka_conf.get('group_id'),
    #     consumer_timeout_ms=kpi_kafka_conf.get('consumer_to') * 1000,
    # )
    kpi_consumer = KafkaConsumer(
        kpi_kafka_conf.get('topic_id'),
        bootstrap_servers=[kafka_server],
        consumer_timeout_ms=kpi_kafka_conf.get('consumer_to') * 1000,
    )
    return kpi_consumer


def init_metric_consumer():
    kafka_server = infer_config.kafka_conf.get('server')
    metric_kafka_conf = infer_config.kafka_conf.get('abnormal_metric_topic')
    # metric_consumer = KafkaConsumer(
    #     metric_kafka_conf.get('topic_id'),
    #     bootstrap_servers=[kafka_server],
    #     group_id=metric_kafka_conf.get('group_id'),
    #     consumer_timeout_ms=metric_kafka_conf.get('consumer_to') * 1000,
    # )
    metric_consumer = KafkaConsumer(
        metric_kafka_conf.get('topic_id'),
        bootstrap_servers=[kafka_server],
        consumer_timeout_ms=metric_kafka_conf.get('consumer_to') * 1000,
    )
    return metric_consumer


def init_cause_producer():
    kafka_server = infer_config.kafka_conf.get('server')
    cause_producer = KafkaProducer(bootstrap_servers=[kafka_server])
    return cause_producer


def init_abn_evt_mgt():
    kpi_consumer = init_kpi_consumer()
    metric_consumer = init_metric_consumer()
    valid_duration = infer_config.infer_conf.get('evt_valid_duration')
    future_duration = infer_config.infer_conf.get('evt_future_duration')
    aging_duration = infer_config.infer_conf.get('evt_aging_duration')
    abn_evt_mgt = AbnEvtMgt(kpi_consumer, metric_consumer, valid_duration=valid_duration,
                            aging_duration=aging_duration, future_duration=future_duration)
    return abn_evt_mgt


def send_cause_event(cause_producer: KafkaProducer, cause_msg):
    logger.logger.debug(json.dumps(cause_msg, indent=2))

    infer_kafka_conf = infer_config.kafka_conf.get('inference_topic')
    try:
        cause_producer.send(infer_kafka_conf.get('topic_id'), json.dumps(cause_msg).encode())
    except KafkaTimeoutError as ex:
        logger.logger.error(ex)
        return
    logger.logger.info('A cause inferring event has been sent to kafka.')


def main():
    pass


if __name__ == "__main__":
    main()
