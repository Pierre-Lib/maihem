MAIHEM - Machine-Automated Identification of Histological lesions for Environmental Monitoring
==============================================================================================
Introduction
------------
A python package using YOLO11 segmentation models to detect and quantify lesions in whole-slides images (WSI),
especially for use in environmental research and monitoring. The package covers both training and
usage of trained models, and allows simultaneous targeting and quantification of multiple lesions. This works builds
on Liboureau et al. 2025 (under review).  
Please see "Quick user guide" further in this file for more information on usage.

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

Quick user guide
----------------
For use 