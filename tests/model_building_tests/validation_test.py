"""Tests for the validation functions"""

import unittest
import shutil
from pathlib import Path

from maihem_code.model_building_tools.validation import validate_model

TEST_DIR = Path(__file__).parent
WEIGHTS_PATH = TEST_DIR / 'model_for_tests' / 'weights'


class MyTestCase(unittest.TestCase):
    """Class to test validation functions"""
    def test_validation(self):
        """Checks that the validation metrics are consistent for
         a same model and validation set"""
        test_validation = validate_model(model_path=WEIGHTS_PATH / 'best.pt',
                                         save_path=TEST_DIR / 'Validation',
                                         save_name='test_validation',
                                         save_tf=True)
        test_map75_box = test_validation['map75_box']
        test_map75_seg = test_validation['map75_seg']
        expected_map75_box = 0.711933287625696
        expected_map75_seg = 0.5280599808858971
        self.assertAlmostEqual(test_map75_box, expected_map75_box, 3)
        self.assertAlmostEqual(test_map75_seg, expected_map75_seg, 3)
        shutil.rmtree(TEST_DIR / 'Validation')
        print("All saved directories removed")


if __name__ == '__main__':
    unittest.main()
