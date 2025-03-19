"""A file to run the examples from maihem package."""

from pathlib import Path
import shutil
from maihem_code.core.main import maihem_main

# Gets the path to the example directory
PATH_TO_HERE = Path(__file__).parent

# Remove all folders with previous results to avoid confusion and buildup
if Path.exists(PATH_TO_HERE / 'test_model'):
    shutil.rmtree('test_model')
if Path.exists(PATH_TO_HERE / 'predict'):
    shutil.rmtree('predict')

# Ask user for input to select the example version
print("Select which example to run (1-4):")
print("1) No input, only checking if the binaries work")
print("2) Training only, input: example_2_input.json")
print("3) Usage only, input: example_3_input.json")
print("4) Training and usage; complete run. Input: example_4_input.json")

# Read user input
example_number = input("Enter a number (1-4):")

# Create a dictionary to match input number to file
example_files_dict = {
    '2': 'example_2_input.json',
    '3': 'example_3_input.json',
    '4': 'example_4_input.json'
}

# Run the package according to the input
if example_number == '1':
    maihem_main()
elif example_number in example_files_dict:
    PATH_TO_INPUT = PATH_TO_HERE / example_files_dict[example_number]
    maihem_main(PATH_TO_INPUT)
else:
    print("Please enter a number between 1 and 4; exiting")
