"""Tests for the package backbone"""

import unittest
import shutil
from pathlib import Path

from maihem_code.backbone.run_functions import (model_training_and_validation,
                                                detections_and_calculations)


TESTS_DIR = Path(__file__).parent.parent

training_instructions = {"dataset": "coco8-seg.yaml",
                         "settings": {
                             "model_architecture": "yolo11n-seg",
                             "model_path": ".",
                             "model_name": "test_model",
                             "validation_name": "test_validation",
                             "save_validation": True
                         },
                         "hyperparameters": {
                             "epochs": 10,
                             "image_size": 640,
                             "batch_size": 16,
                             "device": "cpu",
                             "seed": 1612
                            }
                         }

usage_instructions = {
            "dataset": Path.joinpath(TESTS_DIR, "test_img.jpg"),
            "settings": {
              "model_path": Path.joinpath(TESTS_DIR, "yolo11n-seg.pt"),
              "yaml_path": Path.joinpath(TESTS_DIR,
                                         "coco8-seg_test.yaml"),
              "output_path": ".",
              "save_output": True
            },
            "parameters": {
              "pixel_size": 1,
              "confidence_threshold": 0.25,
              "lesion_names": ["person", "elephant"]
            }
          }


class MyTestCase(unittest.TestCase):
    """A class with test functions for the two
     backbone functions of the package"""

    def test_training_and_validation(self):
        """Tests the training and validation function. Checks that the
         validation metrics are as expected for a specific model training."""
        test_bin_metrics = model_training_and_validation(training_instructions)
        expected_map50_box = 0.9407706548068574
        expected_map75_seg = 0.5277204962825915
        self.assertAlmostEqual(test_bin_metrics['map50_box'],
                               expected_map50_box, 2)
        self.assertAlmostEqual(test_bin_metrics['map75_seg'],
                               expected_map75_seg, 2)
        shutil.rmtree('test_model')
        print("All saved directories removed")

    def test_detections_and_calculations(self):
        """Tests the detection and measurement calculations functions.
        Checks that the final dictionary of measurements obtained for the
        test image is correct."""
        test_merged_detection_metrics = detections_and_calculations(
            usage_instructions
        )
        expected_output = {'person': {
            'test_img.jpg': {
                'count': 1,
                'total area': 4437.090087890625
            }
        },
            'elephant': {
                'test_img.jpg': {
                    'count': 2,
                    'total area': 21790.920776367188
                }
            }
        }
        self.assertAlmostEqual(test_merged_detection_metrics,
                               expected_output, 5)
        shutil.rmtree('predict')
        print("All saved directories removed")


if __name__ == '__main__':
    unittest.main()
