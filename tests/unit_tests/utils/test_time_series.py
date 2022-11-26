import unittest

import pandas as pd

from anteater.core.time_series import TimeSeries


class TestTimeSeries(unittest.TestCase):

    def test_time_series_should_initialized_null_correct(self):
        # given
        metric = 'test_metric'
        labels = {'key': 'test_key', 'label': 'test_label'}
        time_stamps = []
        values = []

        # then
        time_series = TimeSeries(metric, labels, time_stamps, values)

        self.assertEqual(time_series.metric, metric)
        self.assertEqual(time_series.labels, labels)
        self.assertEqual(time_series.time_stamps, time_stamps)
        self.assertEqual(time_series.values, values)

    def test_time_series_should_initializer_correct(self):
        # given
        metric = 'test_metric'
        labels = {'key': 'test_key', 'label': 'test_label'}
        time_stamps = [1667179715, 1667179720, 1667179725, 1667179730, 1667179735]
        values = [4.09, 200, 0, -14.09, 3.00]

        # then
        time_series = TimeSeries(metric, labels, time_stamps, values)

        self.assertEqual(time_series.metric, metric)
        self.assertEqual(time_series.labels, labels)
        self.assertEqual(time_series.time_stamps, time_stamps)
        self.assertEqual(time_series.values, values)

    def test_time_series_should_extend_correct(self):
        # given
        metric = 'test_metric'
        labels = {'key': 'test_key', 'label': 'test_label'}
        time_stamps = [1667179715, 1667179720, 1667179725, 1667179730, 1667179735]
        values = [4.09, 200, 0, -14.09, 3.00]

        # then
        time_series = TimeSeries(metric, labels, time_stamps, values)

        new_time_stamps = [1667179740, 1667179745]
        new_values = [5.01, 6.00]
        time_series.extend(new_time_stamps, new_values)

        self.assertEqual(time_series.metric, metric)
        self.assertEqual(time_series.labels, labels)
        self.assertEqual(time_series.time_stamps, time_stamps + new_time_stamps)
        self.assertEqual(time_series.values, values + new_values)

    def test_time_series_should_to_df_with_null_init(self):
        # given
        metric = 'test_metric'
        labels = {'key': 'test_key', 'label': 'test_label'}
        time_stamps = []
        values = []

        # then
        time_series = TimeSeries(metric, labels, time_stamps, values)

        data_frame = time_series.to_df()

        self.assertEqual(data_frame.shape, (0, ))

    def test_time_series_should_to_df_correct(self):
        # given
        metric = 'test_metric'
        labels = {'key': 'test_key', 'label': 'test_label'}
        time_stamps = [1667179715, 1667179720, 1667179725, 1667179730, 1667179735]
        values = [4.09, 200, 0, -14.09, 3.00]

        # then
        time_series = TimeSeries(metric, labels, time_stamps, values)

        data_frame = time_series.to_df()

        self.assertEqual(data_frame.shape, (5, ))

