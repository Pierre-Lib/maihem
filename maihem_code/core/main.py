# Copyright (c) 2025, Fish Maihem Development
# Distributed under MIT license
"""Perform tasks specified in the input file instructions."""

from maihem_code.core.runner import Runner


def maihem_main(input_file=None):
    """Execute the instructions from the input file.

    Parameters
    ----------
    input_file : str
        The input file to execute
    """
    run = Runner(input_file)
    # error handling in case no input file was provided
    try:
        run.read_instructions()
    except TypeError:
        print("No input file provided; ending run")
    else:
        run.check_instructions()
        run.plan_training()
        run.plan_usage()
        training_results = run.execute_training()
        usage_results = run.execute_usage()
        print(training_results)
        print(usage_results)
