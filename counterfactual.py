from nelder_mead import nelder_mead
import numpy as np
from statsmodels import robust
import gradientgrow as gg


class CounterFactualFinder():
    def __init__(self, clf, data, chosen_attributes=None):
        self.clf = clf
        self.data = data
        self.chosen_attributes = chosen_attributes

    def first_counterfactual_with_nelder_mead(self, instance, target_value=1):
        """

        Args:
            instance:
            target_value:

        Returns:

        """

        def func(x):
            """
            Returns the function to optimize
            X must be np.array
            """

            return target_value - self.clf.predict_proba(x.reshape(1, -1))[0, 1]

        return nelder_mead(func, instance)[0]

    def improved_nelder_mead(self, instance, target_value=1, weigths=None):

        mad = robust.mad(self.data, axis=0)

        def manhattan_distance(y, x=instance, weigths=weigths):

            if weigths is None:
                weigths = np.full(len(x), 1)

            abs = np.abs(x - y)

            result = np.nansum(np.divide(abs, mad) * weigths)
            if np.isinf(result):
                return 0
            return result

        def func(x, l=10):

            feat = l*(target_value - self.clf.predict_proba(x.reshape(1,-1))[0, 1])
            optimize = feat + manhattan_distance(x)

            return optimize

        return nelder_mead(func, instance)[0]

    def get_first_adversarial(self, original_instance):
        """
        Using the GradientSearch approach this method searches the first adversarial
        instance to feed into magnetic_sampling.
        """
        # GradientSearch returns only the two chosen attributes
        dec = gg.Decision(self.data,
                          self.chosen_attributes,
                          original_instance,
                          self.clf)
        # TODO: Automate Parameters
        dec.gradient_search(step=0.05, scale=1.0, nsample=100)
        return dec.get_last_instance()


