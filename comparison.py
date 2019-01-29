"""
Comparing different local post-hoc explanation-by-surrogate approaches

"""

from utils import create_ranges, adjust_features
from sklearn.metrics import accuracy_score

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


def explainer_evaluation(clf, instance, list_of_methods, eval_range_min, eval_range_max, feature_positions):
    """

    All explainers in the list must have the get_explainer method

    Args:
        clf: the original classifier
        instance: the instance for which we applied
        list_of_methods: List of explaination methods
        eval_range_min: Lower left 'corner' of the evaluation rectangle
        eval_range_max: Upper right 'corner' of the evaluation rectangle
        feature_positions: which features of the original instance are considered by the explainers

    Returns:

    """
    exp_pred = []
    eval_data = create_ranges(eval_range_min, eval_range_max, 20)
    grid_data = np.array(np.meshgrid(*eval_data, sparse=False, indexing='ij')).reshape(2, -1).T
    print('grid', grid_data)
    print('grid shape', grid_data.shape)
    eval_data_full = adjust_features(instance, feature_positions, grid_data, 0)
    clf_pred = clf.predict(eval_data_full)

    for method in list_of_methods:
        exp_clf = method.get_explainer()
        exp_pred = exp_clf.predict(eval_data_full[:, feature_positions])
        score = accuracy_score(clf_pred, exp_pred)
        print(method.get_name() + ':' + str(score))

    colors = []
    for y_i in exp_pred:
        if y_i > 0.5:
            colors.append('limegreen')
        else:
            colors.append('tomato')

    plt.scatter(eval_data_full[:, feature_positions[0]], eval_data_full[:, feature_positions[1]], c=colors)
    plt.show()



