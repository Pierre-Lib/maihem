# Copyright (c) 2025, Fish Maihem Development
# Distributed under MIT license
"""Binaries to run the program from the input file."""

import argparse
import datetime
import sys
from maihem_code.core.main import maihem_main

_DATE_FORMAT = '%d.%m.%Y %H:%M:%S'


def hello_world():
    """Greet the user, output program start time and python version."""
    timestart = datetime.datetime.now().strftime(_DATE_FORMAT)
    pyversion = sys.version.split()[0]
    print(f'Start of execution: {timestart}')
    print(f'Python version: {pyversion}')


def parse_input_file():
    """Parse the input file, if specified."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Location of maihem input file',
                        required=False, default=None)
    args_dict = vars(parser.parse_args())
    input_file = args_dict['input']

    return input_file


def bye_world():
    """Output the end time of execution and signal the end of the run."""
    timeend = datetime.datetime.now().strftime(_DATE_FORMAT)
    print(f'End of maihem execution: {timeend}')


def entry_point():
    """Perform entry point function.

    Description
    -----------
    Checks if an input file has been provided, and if so,
    runs the maihem program. Also greets the user before and after the run.
    """
    hello_world()
    input_file = parse_input_file()
    if input_file:
        maihem_main(input_file)
    else:
        print("No input file provided; ending run")
    bye_world()


if __name__ == '__main__':
    entry_point()
