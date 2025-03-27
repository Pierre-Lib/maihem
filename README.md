MAIHEM - Machine-Automated Identification of Histological lesions for Environmental Monitoring
==============================================================================================
Introduction
------------
A python package using YOLO11 segmentation models to detect and quantify lesions in whole-slides images (WSI),
especially for use in environmental research and monitoring. The package covers both training and
usage of trained models, and allows simultaneous targeting and quantification of multiple lesions. This works builds
on Liboureau et al. 2025 (under review).  
Please see "Quick start guide" further in this file for more information on usage.

Installation:
----------------------------------------------------------------
git clone https://github.com/Pierre-Lib/maihem.git

cd maihem

pip install .

Run unit tests (requires pytest):
---------------------------------
pytest

Test quality of code style (requires pylint, pycodestyle and pydocstyle):
-------------------------------------------------------------------------
pylint maihem_code tests example

pycodestyle maihem_code tests example

pydocstyle maihem_code tests example

Test the coverage of unit tests (requires coverage):
----------------------------------------------------
coverage run -m pytest

After execution of the tests:
coverage report -m --ignore-errors

Run examples:
------------
There are four examples showcasing different aspects of the package.  
They can all be accessed through the same method with a simple input step:

cd example  
sh maihem_example.sh

OR

cd example  
python maihem_example.py

Example results will be saved in the example folder, but removed if the example script is
executed again to avoid build-up.

Quick start guide
----------------
Only an input file (JSON format) is needed for use. Maihem can run
either training and validation or detection and segmentation, or both sequentially.  
Make sure the input file is formatted correctly; a template input file
is provided with the package. Note that default values are provided for all parameters
except those with the value "user_input_necessary" in the template file. You can entirely
forgo the "Train" or "Usage" part of the input file to use only one aspect of the package
(as illustrated in example input files 2 and 3).

Once the input file is ready, with the package installed it can be ran from the command line using:  
maihem -i path_to_input_file

Results are saved either as specified in the input file, or directly in the current 
working directory if unspecified.  
Training results include the trained model (.pt format, in the weights folder, includes both the
best and last models trained); a JSON file showing the exact training parameters used;
and validation metrics and graphs.  
Usage results include a JSON file showing the exact usage parameters used;
a JSON file with all detections found in the images along with bounding box and
segmentation coordinates; a JSON file with the measures (number of detected lesions and total area)
of specified lesions for each image); and images showing the segmented areas, assigned lesion names
and confidence of detections.
