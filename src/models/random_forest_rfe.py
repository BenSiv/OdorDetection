"""
random forest model + recursive feature elimination.
"""

# packages
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)


def random_forest_rfe(x_train, y_train, x_test, y_test):
    """random forest model"""
    model = RandomForestClassifier()
    best_accuracy = 0.0
    not_better = 0
    x_train_reduced = x_train.copy()
    x_test_reduced = x_test.copy()
    for num in list(range(2,len(x_train.columns)))[::-1]:
        selector = RFE(model, n_features_to_select=num, step=1)
        selector.fit(x_train_reduced, y_train["label"])
        x_train_reduced = x_train_reduced.loc[:,selector.support_]
        x_test_reduced = x_test_reduced.loc[:,selector.support_]

        model.fit(x_train_reduced, y_train["label"])
        y_pred = model.predict(x_test_reduced)
        report = classification_report(y_test, y_pred, output_dict=True)
        if report["accuracy"] > best_accuracy:
            best_accuracy = report["accuracy"]
            best_model = model
            best_report = report
            best_features = x_train_reduced.columns
            not_better = 0
        else:
            not_better += 1

        print(f"{num} features with accuracy: {report['accuracy']:.0%}")

        # if not_better > 50:
        #     break

    pd.Series(best_features).to_csv(os.path.join(PROJECT_DIR, "data", "stats", "best_features.csv"), index=False)    

    return best_model, best_report
