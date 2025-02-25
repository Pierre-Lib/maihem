
from maihem.core.main import maihem_main
from pathlib import Path

PATH_TO_HERE = Path(__file__).parent
PATH_TO_INPUT = PATH_TO_HERE / 'example_input.json'

maihem_main(PATH_TO_INPUT)

