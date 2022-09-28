import datetime
import json
import os.path
from os.path import dirname, realpath
import unittest
from unittest.mock import Mock, patch, call

from anteater.provider.aom import AomAdapter, create_aom_auth
from anteater.provider.prometheus import PrometheusAdapter
from anteater.provider.base import TimeSeriesProvider


def request_get_mock(*args, **kwargs):
    folder_path = dirname(dirname(dirname(realpath(__file__))))
    file = os.path.join(folder_path, os.sep.join(["data", "request_data.json"]))
    with open(file) as f:
        data = json.load(f)
        return data['metric1'], data['metric2']


class TestTimeSeriesProvider(unittest.TestCase):

    @patch('requests.get')
    def test_should_request_get_json_data_correct(self, request_mock: Mock):
        # given
        metric = request_get_mock()[0]
        response_mock = Mock(status_code=200)
        response_mock.json.return_value = metric
        request_mock.return_value = response_mock

        # then
        data_collector = TimeSeriesProvider(url='http://test')
        result = data_collector.fetch('http://test', params={})

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['metric']['__name__'], 'gala_gopher_cpu_total_used_per')

    @patch('requests.get')
    def test_should_request_get_time_series_data_correct(self, request_mock: Mock):
        # given
        start_time = datetime.datetime(2022, 10, 1, 00, 00)
        end_time = datetime.datetime(2022, 10, 1, 10, 00)

        metric = request_get_mock()[1]
        metric_name = "gala_gopher_cpu_total_used_per1"
        response_mock = Mock(status_code=200)
        response_mock.json.return_value = metric
        request_mock.return_value = response_mock

        # then
        provider = TimeSeriesProvider(url='http://test')
        result = provider.range_query(start_time, end_time, metric_name, query="test")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].metric, metric_name)
        self.assertEqual(len(result[0].time_stamps), 8642)
        self.assertEqual(len(result[0].values), 8642)
        self.assertEqual(result[1].metric, metric_name)
        self.assertEqual(len(result[1].time_stamps), 10)
        self.assertEqual(len(result[1].values), 10)

    @patch('requests.get')
    def test_prometheus_should_request_get_time_series_data_correct(self, request_mock: Mock):
        # given
        start_time = datetime.datetime(2022, 10, 1, 00, 00)
        end_time = datetime.datetime(2022, 10, 1, 10, 00)

        metric = request_get_mock()[1]
        metric_name = "gala_gopher_cpu_total_used_per1"
        response_mock = Mock(status_code=200)
        response_mock.json.return_value = metric
        request_mock.return_value = response_mock

        # then
        provider = PrometheusAdapter(server='server', port="port")
        result = provider.range_query(start_time, end_time, metric_name, query="test")

        url = 'http://server:port/api/v1/query_range'
        calls = [
            call(url, {'query': 'test', 'start': int(start_time.timestamp()), 'end': 1664575200, 'step': 5},
                 headers={}, timeout=30),
            call().json(),
            call(url, {'query': 'test', 'start': 1664575200, 'end': int(end_time.timestamp()), 'step': 5},
                 headers={}, timeout=30),
            call().json()]
        request_mock.assert_has_calls(calls)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].metric, metric_name)
        self.assertEqual(result[1].metric, metric_name)

    @patch('requests.get')
    def test_aom_should_request_get_time_series_data_correct(self, request_mock: Mock):
        # given
        start_time = datetime.datetime(2022, 10, 1, 00, 00)
        end_time = datetime.datetime(2022, 10, 1, 10, 00)

        metric = request_get_mock()[1]
        metric_name = "gala_gopher_cpu_total_used_per1"
        response_mock = Mock(status_code=200)
        response_mock.json.return_value = metric
        request_mock.return_value = response_mock

        # then
        auth_info = {
            'iam_server': 'server',
            'iam_domain': 'domain',
            'iam_user_name': 'user_name',
            'iam_password': 'password',
            'ssl_verify': 0
        }
        aom_auth = create_aom_auth(auth_type='token', auth_info=auth_info)
        provider = AomAdapter(aom_server='http://server', project_id="id", aom_auth=aom_auth)
        result = provider.range_query(start_time, end_time, metric_name, query="test")

        url = 'http://server/v1/id/aom/api/v1/query_range'
        calls = [
            call(url, {'query': 'test', 'start': int(start_time.timestamp()), 'end': 1664575200, 'step': 5},
                 headers={'X-Auth-Token': ''}, timeout=30),
            call().json(),
            call(url, {'query': 'test', 'start': 1664575200, 'end': int(end_time.timestamp()), 'step': 5},
                 headers={'X-Auth-Token': ''}, timeout=30),
            call().json()]
        request_mock.assert_has_calls(calls)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].metric, metric_name)
        self.assertEqual(result[1].metric, metric_name)
