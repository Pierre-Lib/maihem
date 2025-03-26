# Copyright (c) 2025, Fish Maihem Development
# Distributed under MIT license
"""Functions to format output files."""


def merge_detection_measures(count_dict, area_dict):
    """A function to merge the total count and total area dictionaries
    The input dictionaries are created by functions in
     the MeasuresCalculations class:
        count_dict by total_number_of_occurrences
        area_dict by sum_areas_of_class

    Parameters
    ----------
    count_dict : dict
        A dictionary of counts of detections of a specific class in images
    area_dict : dict
        A dictionary of total areas of detections of a specific class in images

    Returns
    -------
    merged_dict : dict
        A dictionary having, for each image, a sub dictionary with both
         count and area
    """

    merged_dict = {
        image: {'count': count_dict[image],
                'total area': area_dict[image]} for image in count_dict
    }
    return merged_dict
