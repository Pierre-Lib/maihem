"""Two functions for use of YOLO models in  environmental histopathology"""

import random
import json

from maihem_code.main.in_out.input import Input
from maihem_code.main.in_out.output import merge_detection_metrics
from maihem_code.main.model_building.training import train_model
from maihem_code.main.model_building.validation import validate_model
from maihem_code.main.model_usage.detections import detection_segmentation
from maihem_code.main.model_usage.metrics_calculations import MetricsCalculations

def model_training_and_validation(dataset_path,
                                  model_architecture = 'yolo11n-seg',
                                  model_path = '.',
                                  model_name = None,
                                  epochs = 100,
                                  batch_size = 16,
                                  image_size = 640,
                                  device = 'cpu',
                                  seed = random.randint(0, 1000),
                                  validation_name = "validation"
                                  ):
    hyperparameters = Input.set_hyperparameters(epochs = epochs,
                                                batch_size = batch_size,
                                                image_size = image_size,
                                                device = device,
                                                seed = seed)
    #Set model name to default if none has been specified
    if model_name is None:
        model_name = f'my_new_{model_architecture}_model'

    print(f"Commencing model training for {epochs} epochs")
    train_model(dataset_path = dataset_path,
                hyperparameters = hyperparameters,
                save_path = model_path,
                save_name = model_name,
                model_architecture = model_architecture)

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
    detections = detection_segmentation(model_path = model_path,
                                        image_path = image_path,
                                        save_path = output_path,
                                        conf_threshold = confidence_threshold)

    metrics_calculator = MetricsCalculations(detections)
    class_names = metrics_calculator.get_class_names(yaml_file_path)
    complete_metrics_dict = {}
    for lesion in lesion_names:
        lesion_count = metrics_calculator.total_number_of_occurrences(class_names, lesion)
        total_lesion_area = metrics_calculator.sum_areas_of_class(class_names, lesion, pixel_size)
        lesion_metrics = merge_detection_metrics(lesion_count, total_lesion_area)
        complete_metrics_dict[lesion] = lesion_metrics

    with open(f'{output_path}/detection.json', 'w', encoding = 'utf-8') as outfile:
        json.dump(complete_metrics_dict, outfile)

    return complete_metrics_dict
