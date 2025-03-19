"""A class to take inputs from a file and format them for use by the functions in the package"""

import datetime
import json
from pathlib import Path
import random
import yaml



class Input:
    """A class that takes inputs and formats them for use in other functions

    ...
    Attributes
    ----------

    Methods
    -------
    parse_instructions_file(input_file)
        Reads the instructions file into a dictionary
    check_path(filename)
        Checks if the provided path exists
    check_lesion_names(lesion_names, yaml_file)
        Checks if the provided lesion names exist in the yaml file
    format_training_instructions(train_instructions)
        Puts the training instructions into a dictionary,
        adding default values for missing instructions
    format_usage_instructions(usage_instructions)
        Puts the usage instructions into a dictionary,
        adding default values for missing instructions
    """

    def __init__(self):
        """
        Parameters
        ----------
        """

    @staticmethod
    def parse_instructions_file(input_file):
        """Reads the instructions file into a dictionary

        Parameters
        ----------
        input_file: str
            Path to the file with instructions

        Returns
        -------
        instructions: dict
            Dictionary of instructions
        """
        with open(input_file, 'r', encoding = 'utf-8') as file:
            instructions = json.load(file)

        return instructions

    @staticmethod
    def check_path(filename):
        """Checks if the provided path exists

        Parameters
        ----------
        filename: str
            Path to file

        Raises
        ------
        ValueError
            If the file cannot be found at the provided path
        """
        filename = Path(filename)
        if not Path.exists(filename):
            raise ValueError(f"File {filename} does not exist!")

    @staticmethod
    def check_lesion_names(lesion_names, yaml_file):
        """Checks if the provided lesion names exist

        Parameters
        ----------
        lesion_names: list
            Lesions to be checked
        yaml_file: str
            Path to the yaml file with lesions the model is trained to detect

        Raises
        ------
        ValueError
            If the lesion name cannot be found in the yaml file
        """
        with open(yaml_file, 'r', encoding = 'utf-8') as file:
            data = yaml.safe_load(file)
        class_names = data.get('names', {})
        for lesion in lesion_names:
            if lesion not in class_names.values():
                raise ValueError(f"Lesion {lesion} does not exist!")

    @staticmethod
    def format_training_instructions(train_instructions):
        """Formats the training instructions into a dictionary. If some instructions are missing,
        replaces them with default values

        Parameters
        train_instructions: dict
            Dictionary of inputted training instructions

        Returns
        -------
        complete_training_instructions: dict
            Dictionary with complete training instructions,
            after merging the inputted and default instructions
        """
        default_settings = {'model_architecture' : 'yolo11n-seg',
                            'model_path' : '.',
                            'model_name' : f'model_created_'
                                           f'{datetime.datetime.now().
                                           strftime("%Y_%m_%d_%H_%M_%S")}',
                            'validation_name' : 'validation',
                            'save_validation' : True
                            }
        default_hyperparameters = {'epochs' : 100,
                                  'batch_size' : 16,
                                  'image_size' : 640,
                                  'device' : 'cpu',
                                  'seed' : random.randint(0, 1000)
                                   }
        complete_training_instructions = {}
        complete_training_instructions['dataset'] = train_instructions['dataset']
        if 'settings' not in train_instructions:
            complete_training_instructions['settings'] = default_settings
        else:
            complete_training_instructions['settings'] = {
                key: train_instructions['settings'].get(key, default) for
                key, default in default_settings.items()}

        if 'hyperparameters' not in train_instructions:
            complete_training_instructions['hyperparameters'] = default_hyperparameters
        else:
            complete_training_instructions['hyperparameters'] = {
                key: train_instructions['hyperparameters'].get(key, default) for
                key, default in default_hyperparameters.items()}

        return complete_training_instructions



    @staticmethod
    def format_usage_instructions(usage_instructions):
        """Formats the usage instructions into a dictionary. If some instructions are missing,
                replaces them with default values

                Parameters
                usage_instructions: dict
                    Dictionary of inputted usage instructions

                Returns
                -------
                complete_usage_instructions: dict
                    Dictionary with complete usage instructions,
                    after merging the inputted and default instructions
                """
        default_settings = {'model_path' : 'wrong_path',
                            'yaml_path' : 'wrong_path',
                            'output_path' : '.',
                            'save_output' : True
                            }
        default_parameters = {'confidence_threshold' : 0.5,
                              'pixel_size' : 1,
                              'lesion_names' : ['not right']
                              }
        complete_usage_instructions = {}
        complete_usage_instructions['dataset'] = usage_instructions['dataset']
        if 'settings' not in usage_instructions:
            complete_usage_instructions['settings'] = default_settings
        else:
            complete_usage_instructions['settings'] = {
                key: usage_instructions['settings'].get(key, default) for
                key, default in default_settings.items()}

        if 'parameters' not in usage_instructions:
            complete_usage_instructions['parameters'] = default_parameters
        else:
            complete_usage_instructions['parameters'] = {
                key: usage_instructions['parameters'].get(key, default) for
                key, default in default_parameters.items()}
        return complete_usage_instructions
