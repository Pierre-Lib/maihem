# Copyright (c) 2025, Fish Maihem Development
# Distributed under MIT license
"""Tests for the training function."""

import unittest
import shutil
from pathlib import Path

from maihem_code.model_building_tools.training import train_model

THIS_DIR = Path(__file__).parent

# Remove any vestigial folders that might mess up the tests
if Path.exists(THIS_DIR / 'test_model'):
    shutil.rmtree(THIS_DIR / 'test_model')


class MyTestCase(unittest.TestCase):
    """Class to test the training function."""

    def test_training(self):
        """Test that the train_model function works consistently."""
        test_hyperparameters = {'epochs': 10,
                                'batch_size': 16,
                                'image_size': 640,
                                'device': 'cpu',
                                'seed': 16}
        test_model_metrics = train_model('coco8-seg.yaml',
                                         test_hyperparameters,
                                         '.',
                                         'test_model',
                                         'yolo11n-seg')
        test_model_seg_map50 = test_model_metrics.seg.map50
        expected_map50 = 0.8488227579919876
        self.assertAlmostEqual(test_model_seg_map50, expected_map50, 1)
        shutil.rmtree('test_model')
        print("All saved directories removed")


if __name__ == '__main__':
    unittest.main()
