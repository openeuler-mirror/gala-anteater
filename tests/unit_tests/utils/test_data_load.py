#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""Unit tests for anteater.utils.data_load."""

import json
import logging
import os
import sys
import tempfile
import unittest
from types import ModuleType

# anteater.utils.data_load imports anteater.utils.log, whose module level
# `logger = Log().get_logger()` reads /etc/gala-anteater/config and raises
# FileNotFoundError on a machine without a deployed environment. Stub the
# log module BEFORE importing data_load so the unit tests run standalone.
if 'anteater.utils.log' not in sys.modules:
    _fake_log = ModuleType('anteater.utils.log')
    _fake_log.logger = logging.getLogger('anteater')
    sys.modules['anteater.utils.log'] = _fake_log

from anteater.utils.data_load import load_job_config, validate_dup


class TempJsonFileMixin:
    """Hands out temporary JSON files that outlive only the current test."""

    def write_json(self, obj):
        """Writes obj to a temporary JSON file and returns its path"""
        return self.write_text(json.dumps(obj))

    def write_text(self, text):
        """Writes text to a temporary file and returns its path

        The file is created inside a TemporaryDirectory whose cleanup is
        registered with addCleanup, so the file is removed when the test
        finishes even if an assertion fails before the caller can delete it.
        """
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        filepath = os.path.join(temp_dir.name, 'job.json')
        with open(filepath, 'w', encoding='utf-8') as f_out:
            f_out.write(text)
        return filepath


class TestValidateDup(unittest.TestCase):

    def test_validate_dup_should_return_true_on_duplicated_metrics(self):
        """A duplicated metric name is reported"""
        self.assertTrue(validate_dup(['a', 'b', 'a']))

    def test_validate_dup_should_return_false_on_unique_metrics(self):
        """Unique metric names are accepted"""
        self.assertFalse(validate_dup(['a', 'b', 'c']))

    def test_validate_dup_should_return_false_on_empty_metrics(self):
        """An empty metric list has no duplicates"""
        self.assertFalse(validate_dup([]))


class TestLoadJobConfig(TempJsonFileMixin, unittest.TestCase):

    def test_load_job_config_should_load_basic_job(self):
        """A minimal job config is parsed into a JobConfig"""
        filepath = self.write_json({
            'name': 'anomaly_detection_test',
            'enable': True,
            'job_type': 'anomaly_detection',
            'detector': 'n_sigma',
            'template': 'default',
            'keywords': ['test'],
            'root_cause_num': 3,
            'kpis': [
                {'metric': 'kpi_metric', 'entity_name': 'machine'},
            ],
            'features': [
                {'metric': 'feature_metric'},
            ],
        })

        job = load_job_config(filepath)

        self.assertEqual(job.name, 'anomaly_detection_test')
        self.assertTrue(job.enable)
        self.assertEqual(job.job_type, 'anomaly_detection')
        self.assertEqual(job.detector, 'n_sigma')
        self.assertEqual(job.root_cause_num, 3)
        self.assertEqual(len(job.kpis), 1)
        self.assertEqual(job.kpis[0].metric, 'kpi_metric')
        self.assertEqual(job.kpis[0].entity_name, 'machine')
        self.assertEqual(len(job.features), 1)
        self.assertEqual(job.features[0].metric, 'feature_metric')

    def test_load_job_config_should_filter_disabled_kpis(self):
        """KPIs with enable=False are filtered out"""
        filepath = self.write_json({
            'name': 'anomaly_detection_test',
            'kpis': [
                {'metric': 'enabled_metric', 'entity_name': 'machine'},
                {'metric': 'disabled_metric', 'entity_name': 'machine',
                 'enable': False},
            ],
        })

        job = load_job_config(filepath)

        self.assertEqual([kpi.metric for kpi in job.kpis], ['enabled_metric'])

    def test_load_job_config_should_raise_on_duplicated_kpi_metric(self):
        """Duplicated kpi metric names raise a ValueError"""
        filepath = self.write_json({
            'name': 'anomaly_detection_test',
            'kpis': [
                {'metric': 'same_metric', 'entity_name': 'machine'},
                {'metric': 'same_metric', 'entity_name': 'machine'},
            ],
        })

        with self.assertRaises(ValueError):
            load_job_config(filepath)

    def test_load_job_config_should_raise_on_duplicated_feature_metric(self):
        """Duplicated feature metric names raise a ValueError"""
        filepath = self.write_json({
            'name': 'anomaly_detection_test',
            'features': [
                {'metric': 'same_metric'},
                {'metric': 'same_metric'},
            ],
        })

        with self.assertRaises(ValueError):
            load_job_config(filepath)

    def test_load_job_config_should_raise_on_invalid_json(self):
        """A malformed JSON file raises a JSONDecodeError"""
        filepath = self.write_text('{invalid json')

        with self.assertRaises(json.JSONDecodeError):
            load_job_config(filepath)

    def test_load_job_config_should_keep_model_config_dict_for_rca(self):
        """A RootCauseAnalysis job keeps model_config as a plain dict"""
        model_config = {'name': 'rca_model', 'params': {'key': 'value'}}
        filepath = self.write_json({
            'name': 'RootCauseAnalysis',
            'model_config': model_config,
        })

        job = load_job_config(filepath)

        self.assertEqual(job.model_config, model_config)


if __name__ == "__main__":
    unittest.main()
