"""Tests for the validation functions"""

import unittest

from maihem_code.main.model_building.validation import validate_model

class MyTestCase(unittest.TestCase):
    """Class to test validation functions"""
    def test_validation(self):
        """Checks that the validation metrics are consistent for a same model and validation set"""
        test_validation = validate_model(model_path= 'test_model/weights/best.pt',
                                         save_path = 'test_model/Validation',
                                         save_name = 'test_validation')
        test_map75_box = test_validation['map75_box']
        test_map75_seg = test_validation['map75_seg']
        expected_map75_box = 0.7050959860383944
        expected_map75_seg = 0.5353737638161722
        self.assertAlmostEqual(test_map75_box, expected_map75_box, 3)
        self.assertAlmostEqual(test_map75_seg, expected_map75_seg, 3)


if __name__ == '__main__':
    unittest.main()
