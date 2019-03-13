import numpy as np

from counterfactual import CounterFactualFinder
from support import SupportFinder
from boundary import BoundaryFinder
from surrogate import LinearSurrogate, TreeSurrogate
import utils

NUM_FEATURES = 2

class Explainer():

    def __init__(self):
        self.cf = None
        self.sp = None
        self.bd = None
        self.sg = None

    def predict(self, X):
        """
        Uses
        Args:
            X:

        Returns:

        """
        pass

    def explain_instance(self, instance, chosen_features=None):
        pass


class DefaultExplainer(Explainer):

    def __init__(self, clf, dataset, chosen_features=None, desired_label=1):
        super().__init__()
        self.clf = clf
        self.dataset = dataset
        self.desired_label = 1

        if chosen_features is None:
            # Use a subset for choosing the significant features
            data_subset = dataset[np.random.randint(dataset.shape[0], size=1000), :]
            labels = self.clf.predict(data_subset)
            self.chosen_features = utils.get_primary_features(data_subset, labels, num_features=NUM_FEATURES)
            print('chosen features: ', self.chosen_features)
        else:
            self.chosen_features = chosen_features

        self.cf = CounterFactualFinder(self.clf, self.dataset, self.chosen_features)
        self.sp = SupportFinder(self.clf, self.dataset, self.chosen_features)
        self.bd = BoundaryFinder(self.clf)
        self.sg = LinearSurrogate(self.clf, self.chosen_features, alpha=0.1)


    def explain_instance(self, instance, chosen_features=None):
        """

        Args:
            instance: to explain
            chosen_features: if different from object specific variable

        Returns:

        """

        self.last_instance = instance

        self.counterfactual = self.cf.first_counterfactual_with_nelder_mead(instance, step=10) + 0.0001
        self.support_set = self.sp.support_points_with_magnetic_sampling(instance, self.counterfactual)
        self.touchpoints = self.bd.touchpoints_using_binary_search(self.support_set, instance, fineness=5)
        self.surrogate = self.sg.train_around_border(self.touchpoints)


