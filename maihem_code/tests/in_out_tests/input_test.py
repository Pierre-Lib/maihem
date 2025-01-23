"""Tests the functions in the input class"""

import unittest

from maihem_code.main.in_out.input import Input

class MyTestCase(unittest.TestCase):
    """Class to test import functions"""
    def test_parameter_setting(self):
        """Tests that the set_hyperparameters functions creates the input dictionary correctly"""
        test_input = Input.set_hyperparameters(epochs = 37,
                                               batch_size = 50,
                                               image_size = 1280,
                                               device = 'cuda',
                                               seed = 12)
        expected_dict = {'epochs': 37,
                         'batch_size': 50,
                         'img_size': 1280,
                         'device': 'cuda',
                         'seed': 12}
        self.assertEqual(test_input, expected_dict)  # add assertion here


if __name__ == '__main__':
    unittest.main()
