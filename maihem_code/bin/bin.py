"""Binaries to run the program from the input file"""

import argparse
import datetime
import sys
from maihem_code.core.main import maihem_main

_DATE_FORMAT = '%d.%m.%Y %H:%M:%S'

def hello_world():
    """Function to greet the user and output the time the program starts and python version"""
    timestart = datetime.datetime.now().strftime(_DATE_FORMAT)
    pyversion = sys.version.split()[0]
    print(f'Start of execution: {timestart}')
    print(f'Python version: {pyversion}')

def parse_input_file():
    """Function to parse the input file, if specified"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Location of maihem input file',
                        required=False, default=None)
    args_dict = vars(parser.parse_args())
    input_file = args_dict['input']

    return input_file

def bye_world():
    """Function to output the end time of execution and signal the end of the run"""
    timeend = datetime.datetime.now().strftime(_DATE_FORMAT)
    print(f'End of maihem execution: {timeend}')


def entry_point():
    """The entry point function that checks if an input file has been provided,
    and if so, runs the maihem program. Also greets the user before and after the run"""
    hello_world()
    input_file = parse_input_file()
    if input_file:
        maihem_main(input_file)
    bye_world()


if __name__ == '__main__':
    entry_point()
