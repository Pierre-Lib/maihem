# Copyright (c) 2025, Fish Maihem Development
# Distributed under MIT license
"""Tests for the Runner class"""

import unittest
from pathlib import Path
import datetime

from maihem_code.core.runner import Runner

TESTS_DIR = Path(__file__).parent.parent
INPUT_FILE_PATH = TESTS_DIR / 'test_input.json'


my_test_runner = Runner(input_file_path=INPUT_FILE_PATH,
                        description="A test runner class")


class MyTestCase(unittest.TestCase):
    """Tests for the Runner class"""
    def test_init(self):
        """Tests that the Runner initialises properly"""
        self.assertEqual(my_test_runner.input_file, INPUT_FILE_PATH)
        self.assertIsInstance(my_test_runner.instructions, dict)
        self.assertIn("test run", my_test_runner.description)

    def test_read_instructions(self):
        """Tests that the instructions are correctly read"""
        my_test_runner.read_instructions()
        number_of_epochs = my_test_runner.instructions[
            "Train"]["hyperparameters"]["epochs"]
        self.assertEqual(number_of_epochs, 10)

    def test_check_instructions(self):
        """Tests that the instructions are correctly checked"""
        my_test_runner.check_instructions()

    def test_plan_training(self):
        """Tests that the training instructions are set correctly"""
        new_test_runner = Runner()
        new_test_runner.instructions['Train'] = {'dataset': 'coco8-seg.yaml',
                                                 'hyperparameters': {
                                                     'seed': 16
                                                 }
                                                 }
        new_test_runner.plan_training()
        expected_instructions = {
            'dataset': 'coco8-seg.yaml',
            'settings': {
                'model_architecture': 'yolo11n-seg',
                'model_path': '.',
                'model_name': f'model_created_{
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
                'validation_name': 'validation',
                'save_validation': True
            },
            'hyperparameters': {
                "epochs": 100,
                "image_size": 640,
                'batch_size': 16,
                "device": "cpu",
                "seed": 16
            }
        }
        self.assertEqual(expected_instructions,
                         new_test_runner.instructions['Train'])

    def test_plan_usage(self):
        """Tests that the usage instructions are set correctly"""
        new_test_runner = Runner()
        new_test_runner.instructions['Usage'] = {'dataset': 'coco8-seg.yaml',
                                                 'settings': {
                                                     'model_path': 'somewhere'
                                                 }
                                                 }
        new_test_runner.plan_usage()
        expected_instructions = {'dataset': 'coco8-seg.yaml',
                                 'settings': {'model_path': 'somewhere',
                                              'output_path': '.',
                                              'save_output': True,
                                              'yaml_path': 'wrong_path'},
                                 'parameters': {'confidence_threshold': 0.5,
                                                'lesion_names': ['not right'],
                                                'pixel_size': 1}
                                 }
        self.assertEqual(expected_instructions,
                         new_test_runner.instructions['Usage'])


if __name__ == '__main__':
    unittest.main()
