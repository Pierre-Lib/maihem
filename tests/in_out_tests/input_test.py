# Copyright (c) 2025, Fish Maihem Development
# Distributed under MIT license
"""Test the functions in the input class."""

import unittest
from pathlib import Path

from maihem_code.in_out.input import Input

TESTS_DIR = Path(__file__).parent.parent

INPUT_FILE_PATH = TESTS_DIR / 'test_input.json'


class MyTestCase(unittest.TestCase):
    """Class to test import functions."""

    def test_parse_instruction_file(self):
        """Test if the instruction file is correctly read into a dictionary."""
        test_input = Input.parse_instructions_file(INPUT_FILE_PATH)
        expected_dict = {
          "Train": {
            "dataset": "coco8-seg.yaml",
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
              "device": "cpu",
              "seed": 16
            }
          },
          "Usage": {
            "dataset": "test_img.jpg",
            "settings": {
              "model_path": "some_path",
              "yaml_path": "some_path",
              "output_path": "some_path",
              "save_output": True
            },
            "parameters": {
              "confidence_threshold": 0.5,
              "lesion_names": ["elephant"]
            }
          }
        }
        self.assertEqual(test_input, expected_dict)

    def test_check_path(self):
        """Test if the check_path function works correctly."""
        example_data_file = TESTS_DIR / 'coco8-seg_test.yaml'
        Input.check_path(example_data_file)
        not_a_file = 'does_not_exist'
        self.assertRaises(ValueError, Input.check_path, not_a_file)

    def test_check_lesion_names(self):
        """Test if the check_lesion_names function works correctly."""
        example_lesion_names = ['elephant', 'moose']
        example_yaml_file = TESTS_DIR / 'coco8-seg_test.yaml'
        Input.check_lesion_names(example_lesion_names, example_yaml_file)
        not_a_lesion = ['ferret']
        self.assertRaises(ValueError, Input.check_lesion_names, not_a_lesion, example_yaml_file)

    def test_format_training_instructions(self):
        """Test if the training instructions get formatted correctly."""
        test_input = Input.parse_instructions_file(INPUT_FILE_PATH)
        test_dict = Input.format_training_instructions(test_input['Train'])
        expected_dict = {
            "dataset": "coco8-seg.yaml",
            "settings": {
              "model_architecture": "yolo11n-seg",
              "model_path": ".",
              "model_name": "test_model",
              "validation_name": "test_validation",
              "save_validation": True
            },
            "hyperparameters": {
              "epochs": 10,
              "batch_size": 16,
              "image_size": 640,
              "device": "cpu",
              "seed": 16
            }
        }
        self.assertEqual(test_dict, expected_dict)

    def test_format_usage_instructions(self):
        """Test if the usage instructions get formatted correctly."""
        test_input = Input.parse_instructions_file(INPUT_FILE_PATH)
        test_dict = Input.format_usage_instructions(test_input['Usage'])
        expected_dict = {"dataset": "test_img.jpg",
                         "settings": {
                             "model_path": "some_path",
                             "yaml_path": "some_path",
                             "output_path": "some_path",
                             "save_output": True
                         },
                         "parameters": {
                             "pixel_size": 1,
                             "confidence_threshold": 0.5,
                             "lesion_names": ["elephant"]
                         }
                         }
        self.assertEqual(test_dict, expected_dict)


if __name__ == '__main__':
    unittest.main()
