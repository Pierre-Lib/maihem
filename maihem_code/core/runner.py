"""A class to perform the various actions from the instruction file and
 run the program"""

from maihem_code.in_out.input import Input
from maihem_code.backbone.run_functions import (model_training_and_validation,
                                                detections_and_calculations)


class Runner:
    """A class with methods to run the different parts of the program

    Attributes
    ----------
    input_file : str
        A string with the path to the input file
    description : str
        A string with the description of the input file
    instructions : dict
        A dictionary with the instructions to run the program

    Methods
    -------
    read_instructions()
        Uses the input functions to read the instruction file into a dictionary
    check_instructions()
        Checks what instructions are given and that the file paths and
         lesion names exist
    plan_training()
        Formats the training instructions to be complete and
         readable by the program
    plan_usage()
        Formats the usage instructions to be complete and
         readable by the program
    execute_training()
        If there are training instructions, runs the training functions
    execute_usage()
        If there are usage instructions, runs the usage functions
        """

    def __init__(self, input_file_path=None,
                 description="YOLO instructions"):
        """
        Parameters
        ----------
        input_file_path : str
            A string with the path to the input file
        description : str
            A string with the description of the input file
        """
        self.input_file = input_file_path
        self.description = description
        self.instructions = {}

    def read_instructions(self):
        """Reads the instructions file into a dictionary"""

        self.instructions = Input.parse_instructions_file(self.input_file)

    def check_instructions(self):
        """Checks what instructions are given and that the file paths and
         lesion names exist"""

        if 'Train' in self.instructions.keys():
            Input.check_path(self.instructions['Train']['dataset'])
        else:
            self.instructions['Train'] = False

        if 'Usage' in self.instructions.keys():
            Input.check_path(self.instructions['Usage']['dataset'])
            Input.check_path(
                self.instructions['Usage']['settings']['yaml_path'])
            if 'Train' not in self.instructions.keys():
                Input.check_path(
                    self.instructions['Usage']['settings']['model_path'])
            Input.check_lesion_names(
                self.instructions['Usage']['parameters']['lesion_names'],
                self.instructions['Usage']['settings']['yaml_path']
            )
        else:
            self.instructions['Usage'] = False

    def plan_training(self):
        """Formats the training instructions to be complete and
         readable by the program"""
        if self.instructions['Train']:
            self.instructions['Train'] = (
                Input.format_training_instructions(self.instructions['Train']))

    def plan_usage(self):
        """Formats the usage instructions to be complete and
         readable by the program"""
        if self.instructions['Usage']:
            self.instructions['Usage'] = Input.format_usage_instructions(
                self.instructions['Usage']
            )

    def execute_training(self):
        """If there are training instructions, runs the training functions

        Returns
        -------
        training_summary : dict
            Dictionary with validation metrics after training
        """

        if not self.instructions["Train"]:
            no_training = "No training instructions; moving to model usage."
            return no_training

        training_summary = model_training_and_validation(
            self.instructions['Train']
        )
        return training_summary

    def execute_usage(self):
        """If there are usage instructions, runs the usage functions

        Returns
        -------
        usage_measures : dict
            Dictionary of all the detections, their number and
             total area for each image"""
        if not self.instructions["Usage"]:
            no_usage = "No usage instructions; saving results and ending run."
            return no_usage

        usage_measures = detections_and_calculations(
            self.instructions['Usage']
        )
        return usage_measures
