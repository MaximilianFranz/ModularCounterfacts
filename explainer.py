from counterfactual import CounterFactualFinder
from support import SupportFinder
from boundary import BoundaryFinder
from surrogate import LinearSurrogate, TreeSurrogate
import utils

NUM_FEATURES = 5

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
            labels = self.clf.predict(self.dataset)
            self.chosen_features = utils.get_primary_features(self.dataset, labels, num_features=NUM_FEATURES)
        else:
            self.chosen_features = chosen_features

        self.cf = CounterFactualFinder(self.clf, self.dataset, chosen_features)
        self.sp = SupportFinder(self.clf, self.dataset, chosen_features)
        self.bd = BoundaryFinder(self.clf)
        self.sg = LinearSurrogate(self.clf, chosen_features, alpha=0.1)


    def explain_instance(self, instance, chosen_features=None):
        """

        Args:
            instance: to explain
            chosen_features: if different from object specific variable

        Returns:

        """

        self.last_instance = instance

        self.counterfactual = self.cf.improved_nelder_mead(instance)
        self.support_set = self.sp.support_points_with_magnetic_sampling(instance, self.counterfactual)
        self.touchpoints = self.bd.touchpoints_using_binary_search(self.support_set, instance, fineness=5)
        self.surrogate = self.sg.train_around_border(self.touchpoints)


