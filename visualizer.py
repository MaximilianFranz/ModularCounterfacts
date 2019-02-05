import numpy as np
from utils import create_ranges, adjust_features

import matplotlib.pyplot as plt
# noinspection PyPackageRequirements
from matplotlib import style
style.use("ggplot")

class ExplanationVisualizer():

    def __init__(self, explainer, features_to_show=None):
        self.explainer = explainer
        if not features_to_show:
            self.features_to_show = self.explainer.features
        else:
            self.features_to_show = features_to_show

        if self.explainer.sg.eval_range is not None:
            self.eval_range = self.explainer.sg.eval_range

    def present_explanation(self, **kwargs):
        """
        Visual presentation of the surrogate

        Args:
            explainer:

        Returns:

        """
        self.plot_results(self.explainer.last_instance, self.features_to_show)

    def plot_results(self, normal_instance, features):
        """
        Plots results of the explanation if all steps have been performed

        Returns: -
        """
        # Works for 2D data only --> len(features) = 2
        if self.explainer.sg.eval_range is not None and len(features) == 2:
            eval_data = create_ranges(self.eval_range[0], self.eval_range[1], 40)
            grid_data = np.array(np.meshgrid(*eval_data, sparse=False, indexing='ij')).reshape(2, -1).T
            eval_data_full = adjust_features(normal_instance, features, grid_data, 0)

            score = self.explainer.score(self.sample_set[:, self.chosen_attributes], self.predictions)
            print('score of explainer ', score)
            exp_pred = self.explainer.sg.surrogate.predict(eval_data_full[:, features].reshape(-1, 1))

            colors = []
            for y_i in exp_pred:
                if y_i > 0.5:
                    colors.append('limegreen')
                else:
                    colors.append('tomato')


            plt.scatter(*normal_instance[features], c='r', marker="X", s=100)
            plt.scatter(grid_data[:, 0], grid_data[:, 1], c=colors, s=10, marker=".")
            plt.show()

