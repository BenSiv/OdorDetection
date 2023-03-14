"""
import all needed functions from src/
"""

# packages
import os
import sys

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args
from load_data import load_data
from log_classification import log_metrics
from summeries_classification import summeries_multiclass_report
from pickle_model import pickle_model

sys.path.append(os.path.join(PROJECT_DIR, "src", "prep"))
from scale import scale
from clip import clip
from derive import derive
from reduce import reduce
from split import split
from fft import fourier

sys.path.append(os.path.join(PROJECT_DIR, "src", "models"))
from most_common import most_common
from random_sampler import random_sampler
from logistic_regression import logistic_regression
from random_forest import random_forest
from gradient_boosting import gradient_boosting
from ada_boost import ada_boost
from dense_neuralnet import dense_neuralnet
from MLP import MLP
from RNN import RNN
from KNN import KNN
from random_forest_rfe import random_forest_rfe