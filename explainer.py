from counterfactual import CounterFactualFinder
from support import SupportFinder
from boundary import BoundaryFinder
from surrogate import LinearSurrogate, TreeSurrogate

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

    def __init__(self, clf, dataset, chosen_features=None):
        self.chosen_features = chosen_features
        self.clf = clf
        self.dataset = dataset



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

        counterfactual = self.cf.first_counterfactual_with_nelder_mead(instance)
        support_set = self.sp.support_points_with_magnetic_sampling(instance, counterfactual)
        touchpoints = self.bd.touchpoints_using_binary_search(support_set, instance, fineness=5)
        surrogate = self.sg.train_around_border(touchpoints)


