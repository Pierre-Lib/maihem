"""Tests for the training function"""

import unittest

from maihem_code.main.model_building.training import train_model

class MyTestCase(unittest.TestCase):
    """Class to test the training function"""
    def test_training(self):
        """Tests that the train_model function works consistently when given
        the same hyperparameters and seed."""
        test_hyperparameters = {'epochs': 20,
                                'batch_size': 16,
                                'img_size': 640,
                                'device': 'cpu',
                                'seed': 16}
        test_model_metrics = train_model('coco8-seg.yaml',
                                         test_hyperparameters,
                                         '.',
                                         'test_model',
                                         'yolo11n-seg')
        test_model_seg_map50 = test_model_metrics.seg.map50
        expected_map50 = 0.8496081399536289
        self.assertAlmostEqual(test_model_seg_map50, expected_map50, 3)  # add assertion here


if __name__ == '__main__':
    unittest.main()
