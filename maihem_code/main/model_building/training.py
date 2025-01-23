"""A function training a YOLO11 segmentation model """
from ultralytics import YOLO

def train_model(dataset_path,
                hyperparameters,
                save_path,
                save_name,
                model_architecture
                ):
    """Trains a YOLO11 segmentation model on a specified dataset

    Parameters
    ----------
    dataset_path : str
        The file location of the dataset
    hyperparameters : dict
        A dictionary containing the hyperparameters of the model including
            epochs (int): number of training epochs
            batch_size (int): batch size for training
            img_size (int): image size for training
            device (str): the device to be used for training (e.g., cpu, gpu)
            seed (int): random seed to be used for training
    save_path: str
        The location to save the trained model
    save_name: str
        The name of the saved model
    model_architecture: str
        The specific YOLO11 architecture to be trained

    Returns
    -------
    Model metrics: ultralytics.utils.metrics.SegmentMetrics
        Model metrics - only for unit testing purposes, model is saved automatically for further use
    """

    #Initialise YOLO model
    model = YOLO(model_architecture)
    #Train the model
    model.train(data = dataset_path,
                epochs = hyperparameters['epochs'],
                batch = hyperparameters['batch_size'],
                imgsz = hyperparameters['img_size'],
                device = hyperparameters['device'],
                seed = hyperparameters['seed'],
                name = save_name,
                project = save_path
                )

    print(f"Training completed. Model saved to {save_name}.pt")
    return model.metrics