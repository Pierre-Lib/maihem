"""A class which takes raw segmentation results and calculates various measures"""
import yaml
import cv2
import numpy

class MeasuresCalculations:
    """
    A class using raw segmentation results from a YOLO11 model to calculate
    the number and area of segmentations, and create heatmaps for them.

    ...
    Attributes
    ----------
    detections : list
        a list of segmentations from an image analysed by a trained YOLO model


    Methods
    -------
    get_class_names
        Gets a dictionary matching the class IDs available in results to actual class names
    get_specific_class_id
        Gets the class ID corresponding to the specified class name
    calculate_area
        Calculates the area of a specific mask, and converts it to real units
         if the pixel conversion ratio was specified
    sum_areas_of_class
        Calculates the total area of all detections of a specific class in each image
    total_number_of_occurrences:
        Calculates the total number of detections of a specific class in each image
    -------

    """

    def __init__(self, detections):
        """
        Parameters
        ----------
        detections : list
            a list of segmentations from an image analysed by a trained YOLO model
        """
        self.detections = detections

    @staticmethod
    def get_class_names(yaml_file_path):
        """
        Parameters
        ----------
        yaml_file_path : str
            Contains the path to a .yaml file with information on classes IDs and names

        Returns
        -------
        class_names : dict
            A dictionary matching class IDs and class names
        """
        with open(yaml_file_path, encoding = 'utf-8') as f:
            data = yaml.safe_load(f)
        class_names = data.get('names', {})
        return class_names

    @staticmethod
    def get_specific_class_id(class_names, specific_class_name):
        """
        Parameters
        ----------
        class_names : dict
            A dictionary matching class IDs and class names
        specific_class_name : str
            The name of a class for which to retrieve the ID

        Returns
        -------
        class_id : int
            The ID corresponding to the specified class name
        """
        class_id = (list(class_names.keys())[list(class_names.values()).index(specific_class_name)])
        return class_id


    @staticmethod
    def calculate_area(mask, image_dimensions, pixel_size):
        """
        Parameters
        ----------
        mask : ultralytics.engine.results.mask
            The binary mask of a detection
        image_dimensions : tuple
            A tuple containing the width and height of the image
        pixel_size : float
            The size of a pixel in real units - units depend on usage

        Returns
        -------
        actual_area : float
            The calculated area of the mask
        """
        mask = mask.numpy()
        mask = cv2.resize(mask, (image_dimensions[0], image_dimensions[1]))
        area_in_pixels = numpy.sum(mask)
        actual_area = area_in_pixels * (pixel_size^2)
        return actual_area

    def sum_areas_of_class(self, class_names, specific_class_name, pixel_size):
        """
        Parameters
        ----------
        class_names : dict
            A dictionary matching class IDs and class names
        specific_class_name : str
            The name of the class for which to calculate the total area
        pixel_size : float
            The size of a pixel in real units - units depend on usage

        Returns
        -------
        class_area_dict : dict
            A dictionary matching image file name and total area of all detections
            for the specified class in the image
        """
        class_area_dict = {}
        for detection in self.detections:
            class_id = self.get_specific_class_id(class_names, specific_class_name)
            image_path = detection.path
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image_dimensions = image.shape
            total_class_area = 0
            for box, mask in zip(detection.boxes, detection.masks.data):
                if int(box.cls) == class_id:
                    mask_area = self.calculate_area(mask, image_dimensions, pixel_size = pixel_size)
                    total_class_area += mask_area
            class_area_dict[image_name] = total_class_area
        return class_area_dict

    def total_number_of_occurrences(self, class_names, specific_class_name):
        """
        Parameters
        ----------
        class_names : dict
            A dictionary mapping class IDs and class names
        specific_class_name : str
            The name of the class for which to calculate the total area

        Returns
        -------
        class_number_dict : dict
            A dictionary matching image file name and total count all detections
            for the specified class in the image
        """
        class_number_dict = {}
        for detection in self.detections:
            class_id = self.get_specific_class_id(class_names, specific_class_name)
            image_name = detection.path.split('/')[-1]
            total_number_of_occurrences = 0
            for box in detection.boxes:
                if int(box.cls) == class_id:
                    total_number_of_occurrences += 1
            class_number_dict[image_name] = total_number_of_occurrences
        return class_number_dict
