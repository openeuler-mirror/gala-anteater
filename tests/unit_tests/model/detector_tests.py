import unittest

from anteater.model.detector.jvm_oom_detector import count_per_minutes


class DetectorTests(unittest.TestCase):

    def test_count_per_minutes_null_correct(self):
        # given
        values1 = []
        values2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        expect = []

        # then
        result1 = count_per_minutes(values1).tolist()
        result2 = count_per_minutes(values2).tolist()

        self.assertEqual(result1, expect)
        self.assertEqual(result2, expect)

    def test_count_per_minutes_correct(self):
        # given
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        expect = [12]

        # then
        result = count_per_minutes(values).tolist()

        self.assertEqual(result, expect)

    def test_count_per_minutes_non_increase_list_correct(self):
        # given
        values = [1, 2, 3, 0, 1, 1, 0, 8, 9, 3, 7, 9, 13, 0, 15, 16]
        expect = [12, 11, 12, 13]

        # then
        result = count_per_minutes(values).tolist()

        self.assertEqual(result, expect)
