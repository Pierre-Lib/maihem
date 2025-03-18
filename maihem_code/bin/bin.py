


import argparse
import datetime
import sys
from maihem_code.core.main import maihem_main

_DATE_FORMAT = '%d.%m.%Y %H:%M:%S'

def hello_world():
    timestart = datetime.datetime.now().strftime(_DATE_FORMAT)
    pyversion = sys.version.split()[0]
    print(f'Start of execution: {timestart}')
    print(f'Python version: {pyversion}')

def parse_input_file():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help=f'Location of maihem input file',
                        required=False, default=None)
    args_dict = vars(parser.parse_args())
    input_file = args_dict['input']

    return input_file

def bye_world():
    """
    Greets.

    """
    timeend = datetime.datetime.now().strftime(_DATE_FORMAT)
    print(f'End of maihem execution: {timeend}')


def entry_point():
    hello_world()
    input_file = parse_input_file()
    if input_file:
        maihem_main(input_file)
    bye_world()


if __name__ == '__main__':
    entry_point()