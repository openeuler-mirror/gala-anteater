import unittest

from anteater.utils.common import to_bytes


class TestCommon(unittest.TestCase):

    def test_to_bytes(self):
        # given
        letter_expects = {
            123456 : 123456,
            '123456': 123456,
            '64b': 64,
            '64k': 64 * 1024,
            '64kb': 64 * 1024,
            '64m': 64 * 1024 * 1024,
            '64mb': 64 * 1024 * 1024,
            '64g': 64 * 1024 * 1024 * 1024,
            '64gb': 64 * 1024 * 1024 * 1024,
            '64K': 64 * 1024,
            '64KB': 64 * 1024,
            '64M': 64 * 1024 * 1024,
            '64MB': 64 * 1024 * 1024,
            '64G': 64 * 1024 * 1024,
            '64GB': 64 * 1024 * 1024,
        }

        # then
        for letter, expect in letter_expects.items():
            result = to_bytes(letter)
            self.assertEqual(result, expect)
