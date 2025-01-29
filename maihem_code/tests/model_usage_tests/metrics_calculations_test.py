"""Tests for the measurement calculations"""

import unittest
import cv2

from maihem_code.main.model_usage.detections import detection_segmentation
from maihem_code.main.model_usage.measures_calculations import MeasuresCalculations

sample_detections = detection_segmentation(
    model_path = '../model_building_tests/test_model/weights/best.pt',
    image_path = '../test_img.jpg',
    save_path = './test_predictions',
    conf_threshold = 0.25
)

#prepare some data to be used by multiple tests
test_metrics = MeasuresCalculations(sample_detections)
test_class_names = MeasuresCalculations.get_class_names(yaml_file_path = '../coco8-seg_test.yaml')
test_image = cv2.imread('../test_img.jpg')
test_image_dimensions = test_image.shape

class MyTestCase(unittest.TestCase):
    """Class to test measurement calculation functions"""
    def test_init(self):
        """Checks that the detection results loaded correctly in the class"""
        self.assertEqual(len(test_metrics.detections[0]), 3)

    def test_get_class_names(self):
        """Checks that the function gets the correct IDs : names combinations"""
        self.assertEqual(test_class_names[16], 'dog')
        self.assertEqual(test_class_names[53], 'pizza')

    def test_get_specific_class_id(self):
        """Checks that the function correctly identifies the class ID from its name"""
        test_id = MeasuresCalculations.get_specific_class_id(test_class_names, 'moose')
        self.assertEqual(test_id, 47)

    def test_calculate_area(self):
        """Checks that the function correctly calculates the area of a detection"""
        test_mask = test_metrics.detections[0].masks.data[2]
        test_area = MeasuresCalculations.calculate_area(test_mask, test_image_dimensions, 1)
        expected_area = 4708.871337890625
        self.assertAlmostEqual(test_area, expected_area, 1)

    def test_calculate_sum_areas_of_class(self):
        """Checks that the function correctly calculates the sum of areas for a given class"""
        test_sum_of_areas = test_metrics.sum_areas_of_class(test_class_names, 'elephant', 1)
        expected_sum_of_areas = 21949.565185546875
        self.assertAlmostEqual(test_sum_of_areas['test_img.jpg'], expected_sum_of_areas, 1)

    def test_total_number_of_occurrences(self):
        """Checks that the functions correctly counts the number of occurrences of a given class"""
        test_sum_of_occurrences = test_metrics.total_number_of_occurrences(test_class_names,
                                                                           'elephant')
        self.assertEqual(test_sum_of_occurrences['test_img.jpg'],2)



if __name__ == '__main__':
    unittest.main()
