"""Two functions for use of YOLO models in  environmental histopathology"""

import random
import json

from maihem_code.main.in_out.input import Input
from maihem_code.main.in_out.output import merge_detection_measures
from maihem_code.main.model_building.training import train_model
from maihem_code.main.model_building.validation import validate_model
from maihem_code.main.model_usage.detections import detection_segmentation
from maihem_code.main.model_usage.measures_calculations import MeasuresCalculations

def model_training_and_validation(dataset_path,
                                  model_architecture = 'yolo11n-seg',
                                  model_path = '.',
                                  model_name = None,
                                  epochs = 100,
                                  batch_size = 16,
                                  image_size = 640,
                                  device = 'cpu',
                                  seed = random.randint(0, 1000),
                                  validation_name = 'validation'
                                  ):
    """A function to train a model, save it, run a validation and output validation metrics.

    Parameters
    ----------
    dataset_path : str
        Location of the dataset .yaml file for training and validation
    model_architecture : str
        Type of YOLO11 segmentation model to be trained
    model_path : str
        Location to save the trained model
    model_name : str
        Name to give the trained model
    epochs : int
        Number of epochs to train the model
    batch_size : int
        Batch size for training
    image_size : int
        Size of images for training
    device : str
        Device to use for training
    seed : int
        Random seed for training
    validation_name : str
        Name to give the saved validation data

    Returns
    -------
    validation_metrics : dict
        Dictionary of bounding box and segmentation validation metrics
    """
    #format the hyperparameters as needed for input
    hyperparameters = Input.set_hyperparameters(epochs = epochs,
                                                batch_size = batch_size,
                                                image_size = image_size,
                                                device = device,
                                                seed = seed)

    #Set model name to default if none has been specified
    if model_name is None:
        model_name = f'my_new_{model_architecture}_model'

    #Train the model
    print(f"Commencing model training for {epochs} epochs")
    train_model(dataset_path = dataset_path,
                hyperparameters = hyperparameters,
                save_path = model_path,
                save_name = model_name,
                model_architecture = model_architecture)

    #Validating the model
    print(f"Model training completed. Model weights save in\n"
          f"{model_path}/{model_name}/weights/best.pt\nCommencing model validation")
    validation_metrics = validate_model(model_path = f'{model_path}/{model_name}/weights/best.pt',
                                        save_path = f'{model_path}/{model_name}/Validation',
                                        save_name = validation_name,)
    print(f"Model validation metrics saved in\n"
          f"{model_path}/{model_name}/Validation/{validation_name}")
    return validation_metrics


def detections_and_calculations(model_path,
                                yaml_file_path,
                                image_path,
                                output_path = '.',
                                confidence_threshold = 0.25,
                                lesion_names = None,
                                pixel_size = 1):
    """A function to detect objects in one or more images, and calculate the total number
    of objects and summed area of these objects for each specified object class.

    Parameters
    ----------
    model_path : str
        Location of the model to use
    yaml_file_path : str
        Location of the yaml file for the training dataset, to extract class names and IDs
    image_path : str
        Location of the images to use
    output_path : str
        Location to save the output
    confidence_threshold : float
        Confidence threshold for detections
    lesion_names : list
        Specific classes for which to perform calculations
    pixel_size : float
        Size of pixels to calculate real area of detections

    Returns
    -------
    complete_measures_dict : dict
        Dictionary of
    """
    detections = detection_segmentation(model_path = model_path,
                                        image_path = image_path,
                                        save_path = output_path,
                                        conf_threshold = confidence_threshold)

    measure_calculator = MeasuresCalculations(detections)
    class_names = measure_calculator.get_class_names(yaml_file_path)
    complete_measures_dict = {}
    for lesion in lesion_names:
        lesion_count = measure_calculator.total_number_of_occurrences(class_names, lesion)
        total_lesion_area = measure_calculator.sum_areas_of_class(class_names, lesion, pixel_size)
        lesion_measures = merge_detection_measures(lesion_count, total_lesion_area)
        complete_measures_dict[lesion] = lesion_measures

    with open(f'{output_path}/predict/measures.json', 'w', encoding = 'utf-8') as outfile:
        json.dump(complete_measures_dict, outfile, indent = 4)

    return complete_measures_dict
