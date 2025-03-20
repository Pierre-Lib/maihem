"""Tests for the validation functions"""

import unittest
import shutil
from pathlib import Path

from maihem_code.model_building_tools.training import train_model
from maihem_code.model_building_tools.validation import validate_model

TEST_DIR = Path(__file__).parent
WEIGHTS_PATH = TEST_DIR / 'dummy_model' / 'weights'

#Make a model to use for validation
dummy_hyperparameters = {'epochs': 1,
                         'batch_size': 16,
                         'image_size': 640,
                         'device': 'cpu',
                         'seed': 1612}


class MyTestCase(unittest.TestCase):
    """Class to test validation functions"""
    def test_validation(self):
        """Checks that the validation metrics are consistent for
         a same model and validation set"""
        train_model('coco8-seg.yaml',
                    dummy_hyperparameters,
                    '.',
                    'dummy_model',
                    'yolo11n-seg')
        test_validation = validate_model(model_path=WEIGHTS_PATH / 'best.pt',
                                         save_path=TEST_DIR / 'Validation',
                                         save_name='test_validation',
                                         save_tf=True)
        test_map75_box = test_validation['map75_box']
        test_map75_seg = test_validation['map75_seg']
        expected_map75_box = 0.6952742421530016
        expected_map75_seg = 0.5142414070891514
        self.assertAlmostEqual(test_map75_box, expected_map75_box, 3)
        self.assertAlmostEqual(test_map75_seg, expected_map75_seg, 3)
        shutil.rmtree(TEST_DIR / 'Validation')
        #shutil.rmtree(TEST_DIR / 'dummy_model')
        print("All saved directories removed")


if __name__ == '__main__':
    unittest.main()
