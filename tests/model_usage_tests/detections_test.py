# Copyright (c) 2025, Fish Maihem Development
# Distributed under MIT license
"""Tests for detection functions"""
import unittest
import shutil
from pathlib import Path

from maihem_code.model_usage_tools.detections import detection_segmentation

THIS_DIR = Path(__file__).parent
TESTS_DIR = THIS_DIR.parent


class MyTestCase(unittest.TestCase):
    """Class to test detection functions"""
    def test_detections(self):
        """tests that the class and one of the coordinates of a test detection
         are as expected"""
        test_detection = detection_segmentation(
            model_path=TESTS_DIR / 'yolo11n-seg.pt',
            image_path=TESTS_DIR / 'test_img.jpg',
            save_path=THIS_DIR / 'test_predictions',
            conf_threshold=0.25
        )
        detection = test_detection[0]
        box = detection.boxes[0]
        class_id = int(box.cls)
        bbox = box.xyxy.cpu().numpy()[0]
        bbox_0 = bbox[0]
        test_dict = {'class': class_id,
                     'box_0': bbox_0}
        expected_dict = {'class': 20,
                         'box_0': 241.86955}

        self.assertEqual(test_dict['class'], expected_dict['class'])
        self.assertAlmostEqual(test_dict['box_0'], expected_dict['box_0'], 1)
        shutil.rmtree(THIS_DIR / 'test_predictions')
        print("All saved directories removed")


if __name__ == '__main__':
    unittest.main()
