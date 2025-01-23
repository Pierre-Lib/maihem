import unittest
import cv2

from maihem_code.main.model_usage.detections import detection_segmentation
from maihem_code.main.model_usage.measures_calculations import MetricsCalculations

sample_detections = detection_segmentation(model_path = '../model_building_tests/test_model/weights/best.pt',
                                        image_path = 'test_img.jpg',
                                        save_path = './test_predictions',
                                        conf_threshold = 0.25)

test_metrics = MetricsCalculations(sample_detections)
test_class_names = MetricsCalculations.get_class_names(yaml_file_path = 'coco8-seg_test.yaml')
test_image = cv2.imread('test_img.jpg')
test_image_dimensions = test_image.shape

class MyTestCase(unittest.TestCase):
    def test_init(self):
        self.assertEqual(len(test_metrics.detections[0]), 3)

    def test_get_class_names(self):
        self.assertEqual(test_class_names[16], 'dog')
        self.assertEqual(test_class_names[53], 'pizza')

    def test_get_specific_class_id(self):
        test_id = MetricsCalculations.get_specific_class_id(test_class_names, 'moose')
        self.assertEqual(test_id, 47)

    def test_calculate_area(self):
        test_mask = test_metrics.detections[0].masks.data[2]
        test_area = MetricsCalculations.calculate_area(test_mask, test_image_dimensions, 1)
        self.assertEqual(test_area,4708.871337890625)

    def test_calculate_sum_areas_of_class(self):
        test_sum_of_areas = test_metrics.sum_areas_of_class(test_class_names, 'elephant', 1)
        self.assertEqual(test_sum_of_areas['test_img.jpg'],21949.565185546875)

    def test_total_number_of_occurrences(self):
        test_sum_of_occurrences = test_metrics.total_number_of_occurrences(test_class_names, 'elephant')
        self.assertEqual(test_sum_of_occurrences['test_img.jpg'],2)



if __name__ == '__main__':
    unittest.main()
