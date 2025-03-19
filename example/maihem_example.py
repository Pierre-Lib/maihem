
from maihem_code.core.main import maihem_main
from pathlib import Path
import shutil

PATH_TO_HERE = Path(__file__).parent
PATH_TO_INPUT = PATH_TO_HERE / 'example_4_input.json'

if Path.exists(PATH_TO_HERE / 'test_model'):
    shutil.rmtree('test_model')
if Path.exists(PATH_TO_HERE / 'predict'):
    shutil.rmtree('predict')

maihem_main(PATH_TO_INPUT)
