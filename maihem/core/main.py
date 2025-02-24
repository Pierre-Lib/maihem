from maihem.core.runner import Runner


def maihem_main(input_file = None):
    run = Runner(input_file)
    run.read_instructions()
    run.check_instructions()
    run.plan_training()
    run.plan_usage()
    training_results = run.execute_training()
    usage_results = run.execute_usage
    print(training_results)
    print(usage_results)
