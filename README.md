MAIHEM - Machine-Automated Identification of Histological lesions for Environmental Monitoring
----------------------------------------------------------------------------------------------

Installation:
----------------------------------------------------------------
git clone https://github.com/Pierre-Lib/maihem.git

pip install .

Run unit tests (requires pytest):
---------------------------------
pytest

Test quality of code style (requires pylint and pycodestyle):
-------------------------------------------------------------
pylint maihem_code example tests

pycodestyle */*.py

Run examples:
------------
There are four examples showcasing different aspects of the package.\
They can all be accessed through the same method with a simple input step:

cd example\
sh maihem_example.sh

OR

cd example\
python maihem_example.py
