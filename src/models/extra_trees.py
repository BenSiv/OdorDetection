"""
random forest model
"""

# packages
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

def extra_trees(x_train, y_train, x_test, y_test):
    """random forest model"""
    model = ExtraTreesClassifier()
    model.fit(x_train, y_train["label"])

    y_pred = model.predict(x_test)
    report = classification_report(y_test["label"], y_pred, output_dict=True)

    return model, report
