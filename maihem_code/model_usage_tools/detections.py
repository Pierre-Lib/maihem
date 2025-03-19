"""A function which uses the trained model to detect objects in a new image
and creates a file with segmentation data for further quantitative analysis"""

import json
from ultralytics import YOLO


def detection_segmentation(model_path, image_path, save_path, conf_threshold):
    """Use the trained model to detect and segment objects in a new image

    Parameters
    ----------
    model_path : str
        Path to trained model
    image_path : str
        Path to image
    save_path : str
        Path to output json file
    conf_threshold : float
        Confidence threshold for object detection

    Returns
    -------
    detection_results: list
        A list of detections for each image analysed.
        Each image result is in format ultralytics.engine.results.Results
    """

    model = YOLO(model_path)

    detection_results = model.predict(source=image_path,
                                      conf=conf_threshold,
                                      project=save_path,
                                      save=True)

    all_detections_data = []
    for image_result in detection_results:
        image_path = image_result.path
        if len(image_result.boxes.cls) == 0:
            print(f"No detections in image {image_path}")
            detection_data = {
                'image_path': image_path,
                'class_id': 'None'
            }
            all_detections_data.append(detection_data)
            continue
        for box, mask in zip(image_result.boxes, image_result.masks):
            class_id = int(box.cls)
            bounding_box = box.xyxy.cpu().numpy() if hasattr(
                box.xyxy, 'cpu') else box.xyxy
            bounding_box = bounding_box[0]
            mask_array = mask.xy[0].tolist()
            detection = {
                'image_path': image_path,
                'class_id': class_id,
                'confidence': float(box.conf),
                'bbox': [
                    int(bounding_box[0]),
                    int(bounding_box[1]),
                    int(bounding_box[2]),
                    int(bounding_box[3])
                ],
                'segmentation': mask_array
                }
            all_detections_data.append(detection)

    with open(f'{save_path}/predict/detections.json', 'w',
              encoding='utf-8') as outfile:
        json.dump(all_detections_data, outfile, indent=4)

    return detection_results
