"""A class to take inputs and format them for use by the functions in the package"""


class Input:
    """A class that takes inputs and formats them for use in other functions

    ...
    Attributes
    ----------

    Methods
    -------
    set_hyperparameters
        Takes all hyperparameters as single inputs and returns them as a dictionary for use
        in training.py model_training
    """

    def __init__(self):
        """
        Parameters
        ----------
        """


    @staticmethod
    def set_hyperparameters(epochs, batch_size, image_size, device, seed):
        """
        Parameters
        ----------
        epochs: int
            The number of epochs to train the model
        batch_size: int
            The batch size for training
        image_size: int
            The size of the images used for training
        device: str
            The device on which to train the model
        seed: int
            The random seed for model training

        Returns
        -------
        hyperparameters: dict
            Dictionary containing the hyperparameters of the model
        """
        hyperparameters_dict = {
            'epochs': epochs,
            'batch_size': batch_size,
            'img_size': image_size,
            'device': device,
            'seed': seed
        }
        return hyperparameters_dict
