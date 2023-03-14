"""
run best model
"""

# packages
import os
import sys
import pickle
import numpy as np

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from load_data import load_data

def main():
    best_model_path = os.path.join(PROJECT_DIR, "src", "best_model.pkl")

    best_model_file = open(best_model_path,'rb')
    model = pickle.load(best_model_file)

    labels, features = load_data(PROJECT_DIR)
    
    random_index = np.random.choice(features.shape[0])
    rand_features = features.iloc[random_index,:]
    rand_label = labels.iloc[random_index,:]
    
    model.predict(rand_features.values)
