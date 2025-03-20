"""Two functions for use of YOLO models in  environmental histopathology"""
import json
from maihem_code.in_out.output import merge_detection_measures
from maihem_code.model_building_tools.training import train_model
from maihem_code.model_building_tools.validation import validate_model
from maihem_code.model_usage_tools.detections import detection_segmentation
from maihem_code.model_usage_tools.measures_calculations import MeasuresCalculations


def model_training_and_validation(training_instructions):
    """A function to train a model, save it,
     run a validation and output validation metrics.

    Parameters
    ----------
    training_instructions: dict
        Dictionary with instructions for training the model:
            dataset: path to the training dataset
            hyperparameters: dictionary with hyperparameters
            model_path: path to save the trained model
            model_name: name of the trained model
            model_architecture: architecture of the trained model
            validation_name: name of the folder to save the validation metrics
            save_validation: True/False to save the validation metrics

    Returns
    -------
    validation_metrics : dict
        Dictionary of bounding box and segmentation validation metrics
    """

    # Assign values
    data_path = training_instructions['dataset']
    model_hyperparameters = training_instructions['hyperparameters']
    saving_path = training_instructions['settings']['model_path']
    saving_name = training_instructions['settings']['model_name']
    model_arch = training_instructions['settings']['model_architecture']
    saving_validation = training_instructions['settings']['validation_name']
    save_tf = training_instructions['settings']['save_validation']

    # Train the model
    print(f"Commencing model training for"
          f" {model_hyperparameters['epochs']} epochs")
    train_model(dataset_path=data_path,
                hyperparameters=model_hyperparameters,
                save_path=saving_path,
                save_name=saving_name,
                model_architecture=model_arch
                )

    # Validating the model
    print(f"Model training completed. Model weights save in\n"
          f"{saving_path}/{saving_name}/weights/best.pt\n"
          f"Commencing model validation")
    validation_metrics = validate_model(
        model_path=f'{saving_path}/{saving_name}/weights/best.pt',
        save_path=f'{saving_path}/{saving_name}/Validation',
        save_name=saving_validation,
        save_tf=save_tf
    )

    print(f"Model validation metrics saved in\n"
          f"{saving_path}/{saving_name}/Validation/"
          f"{saving_validation}/metrics.json")
    return validation_metrics


def detections_and_calculations(usage_instructions):
    """A function to detect objects in one or more images,
    and calculate the total number of objects and summed area
    of these objects for each specified object class.

    Parameters
    ----------
    usage_instructions: dict
        A dictionary with instructions for detection and calculations
            model_path: path to the trained model to be used
            dataset: path to the dataset to be analysed
            yaml_path: path to the yaml file with lesion names and IDs
            output_path: path to save the data
            confidence_threshold: confidence threshold for detections
            lesion_names: names of the lesions to be detected
            pixel_size: size of pixels (in μm)

    Returns
    -------
    complete_measures_dict : dict
        Dictionary of all the detections, their number and
         total area for each image
    """

    # Assign values
    path_to_model = usage_instructions['settings']['model_path']
    path_to_data = usage_instructions['dataset']
    path_to_yaml = usage_instructions['settings']['yaml_path']
    saving_path = usage_instructions['settings']['output_path']
    confidence_threshold = usage_instructions[
        'parameters']['confidence_threshold']
    lesion_names = usage_instructions['parameters']['lesion_names']
    pixel_size = usage_instructions['parameters']['pixel_size']

    detections = detection_segmentation(model_path=path_to_model,
                                        image_path=path_to_data,
                                        save_path=saving_path,
                                        conf_threshold=confidence_threshold)

    measure_calculator = MeasuresCalculations(detections)
    class_names = measure_calculator.get_class_names(path_to_yaml)
    complete_measures_dict = {}
    for lesion in lesion_names:
        lesion_count = measure_calculator.total_number_of_occurrences(
            class_names, lesion)
        total_lesion_area = measure_calculator.sum_areas_of_class(
            class_names, lesion, pixel_size)
        lesion_measures = merge_detection_measures(
            lesion_count, total_lesion_area)
        complete_measures_dict[lesion] = lesion_measures

    with open(f'{saving_path}/predict/measures.json',
              'w', encoding='utf-8') as outfile:
        json.dump(complete_measures_dict, outfile, indent=4)

    return complete_measures_dict
