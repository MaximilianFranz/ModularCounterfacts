import numpy as np

from counterfactual import CounterFactualFinder
from support import SupportFinder
from boundary import BoundaryFinder
from surrogate import LinearSurrogate, TreeSurrogate
import utils

NUM_FEATURES = 10

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

    def __init__(self, clf, dataset, chosen_features=None, num_features=NUM_FEATURES, features_names=None, testset=None, testlabels=None):
        super().__init__()
        self.clf = clf
        self.dataset = dataset
        self.num_features = num_features # used when no features chosen
        self.feature_names = np.array(features_names)
        self.testset = testset
        self.testlabels = np.array(testlabels)

        if chosen_features is None:
            # Use a subset for choosing the significant features
            data_subset = dataset[np.random.randint(dataset.shape[0], size=7000), :]
            labels = self.clf.predict(data_subset)
            self.chosen_features = utils.get_primary_features(data_subset, labels, num_features=self.num_features)
            print('chosen features: ', self.feature_names[self.chosen_features])
        else:
            self.chosen_features = chosen_features

        self.cf = CounterFactualFinder(self.clf, self.dataset, self.chosen_features)
        self.sp = SupportFinder(self.clf, self.dataset, self.chosen_features)
        self.bd = BoundaryFinder(self.clf)
        self.sg = LinearSurrogate(self.clf, self.chosen_features, self.dataset, alpha=0.0001)


    def explain_instance(self, instance, chosen_features=None, target_label=1):
        """

        Args:
            instance: to explain
            chosen_features: if different from object specific variable

        Returns:

        """

        self.last_instance = instance

        self.counterfactual = self.cf.improved_nelder_mead(instance, target_value=target_label, step=10)
        self.support_set = self.sp.support_points_with_magnetic_sampling(instance, self.counterfactual)
        self.touchpoints = self.bd.touchpoints_using_binary_search(self.support_set, instance, fineness=5)
        self.surrogate = self.sg.train_around_border(self.touchpoints)



    def get_counterfactuals(self, instances):

        results = []
        i = 0

        for instance in instances:
            print(i)
            counter = self.cf.improved_nelder_mead(instance, step=1)
            results.append(counter)
            i += 1

        return results