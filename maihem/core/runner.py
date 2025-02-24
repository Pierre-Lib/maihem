from pickle import FALSE

from maihem.in_out.input import Input
from maihem.backbone.run_functions import model_training_and_validation, detections_and_calculations

class Runner:
    def __init__(self, input_file = None, description = "YOLO instructions"):
        self.input_file = input_file
        self.description = description
        self.instructions = dict()

    def read_instructions(self):
        self.instructions = Input.parse_instructions_file(self.input_file)

    def check_instructions(self):
        if 'Train' in self.instructions.keys():
            Input.check_path(self.instructions['Train']['dataset'])
        else:
            self.instructions['Train'] = False

        if 'Usage' in self.instructions.keys():
            Input.check_path(self.instructions['Usage']['dataset'])
            Input.check_path(self.instructions['Usage']['settings']['model_path'])
            Input.check_path(self.instructions['Usage']['settings']['yaml_path'])
            Input.check_lesion_names(self.instructions['Usage']['parameters']['lesion_names'], self.instructions['Usage']['settings']['yaml_path'])
        else:
            self.instructions['Usage'] = False

    def plan_training(self):
        if not self.instructions['Train']:
            self.instructions['Train'] = Input.format_training_instructions(self.instructions['Train'])


    def plan_usage(self):
        if not self.instructions['Usage']:
            self.instructions['Usage'] = Input.format_usage_instructions(self.instructions['Usage'])

    def execute_training(self):
        if not self.instructions["Train"]:
            print("No training instructions; moving to model usage.")
        else:
            training_summary = model_training_and_validation(self.instructions['Train'])
            return training_summary

    def execute_usage(self):
        if not self.instructions["Usage"]:
            print("No usage instructions; saving results and ending run.")
        else:
            usage_measures = detections_and_calculations(self.instructions['Usage'])
            return usage_measures