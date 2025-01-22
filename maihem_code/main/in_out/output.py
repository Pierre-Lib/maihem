"""Functions to format output files."""

def merge_detection_metrics(count_dict, area_dict):
    """A function to merge the total count and total area dictionaries
    The input dictionaries are created by functions in the MetricsCalculations class:
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
    merged : dict
        A dictionary having, for each image, a sub dictionary with both count and area
    """
    merged_dic = {}
    for key in count_dict:
        merged_dic[key] = {'count': count_dict[key], 'total area': area_dict[key]}

    return merged_dic
