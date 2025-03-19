"""Goes through the input file instructions and uses them to perform the designated tasks"""

from maihem_code.core.runner import Runner


def maihem_main(input_file = None):
    """Function to execute the instructions from the input file

    Parameters
    ----------
    input_file : str
        The input file to execute
    """
    if not input_file:
        raise TypeError("No input file provided; ending run")
    run = Runner(input_file)
    run.read_instructions()
    run.check_instructions()
    run.plan_training()
    run.plan_usage()
    training_results = run.execute_training()
    usage_results = run.execute_usage()
    print(training_results)
    print(usage_results)
