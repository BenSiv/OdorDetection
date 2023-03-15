"""
runs all model and preprocessing combinations
"""

# packages
import os
import itertools

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def run_tests(models, preps):
    combinations = []
    for p in range(len(preps)+1):
        for combination in itertools.combinations(preps, p):
            combinations.append(combination)

    for model in models:
        for combo in combinations:
            if combo:
                command = f"""python -W ignore {os.path.join(PROJECT_DIR, "main.py")} --model {model} --{" --".join(combo)}"""
            else:
                command = f"""python -W ignore {os.path.join(PROJECT_DIR, "main.py")} --model {model}"""
            print(command)
            os.system(command)

if __name__ == "__main__":
    # preps = ["scale", "clip", "derive", "reduce"]
    preps = []

    # models = [
    #     "random_sampler", "logistic_regression", "random_forest",
    #     "gradient_boosting", "ada_boost", "dense_neuralnet",
    #     "MLP", "RNN", "KNN"
    # ]

    # models = [
    #     "logistic_regression", "random_forest", "gradient_boosting", "MLP"
    # ]
    
    models = ["random_forest"]
        
    run_tests(models, preps)
