"""A function to obtain validation metrics for the trained model"""

from ultralytics import YOLO

def validate_model(model_path, save_path, save_name, save_tf):
    """Runs validation on the previously trained model and outputs metrics
    Only the path to the trained model is needed, other arguments are remembered
    For now, default validation metrics are used

    Parameters
    ----------
    model_path : str
        Path to trained model
    save_path : str
        Path to save validation metrics
    save_name : str
        Name to save validation metrics
    save_tf : bool
        Whether to save validation metrics

    Returns
    -------
    metrics : dict
        Dictionary of validation metrics including mean average precision (mAP)
        for both boxes and segmentation at various intersection over union (IoU)
        thresholds.
    """

    model = YOLO(model_path)

    validation = model.val(plots = True, project = save_path, name = save_name, save_json = save_tf)

    metrics = {'map50-95_box' : validation.box.map,
               'map50_box': validation.box.map50,
               'map75_box' : validation.box.map75,
               'map50-95_seg' : validation.seg.map,
               'map50_seg' : validation.seg.map50,
               'map75_seg' : validation.seg.map75}

    return metrics
