"""
TODO: Make subclasses for linear surrogate / tree surrogate / decision rules / etc.
"""
import numpy as np
from sklearn.linear_model import RidgeClassifier, lars_path
from sklearn.tree import DecisionTreeClassifier

from graph_export import export_tree

class SurrogateTrainer():

    def __init__(self, clf, chosen_features):
        """

        Args:
            chosen_features:
        """
        self.chosen_features = chosen_features
        self.clf = clf

        self.eval_range = None # Later holds the range of points sampled and thus a approximation of the eval area of the explainer
        self.surrogate = None # Last trained explainer is available here

    def sample_normal(self, border_touchpoints, num_samples, sigma):
        """
        Samples around the border_touchpoints with a normal distribution to generate
        a dataset for training a linear model which yields the explanation

        Normal distribution is parametrized based on the distribution of
        the border_touchpoints, so that we sample along the decision boundary

        """
        max_arg = np.amax(border_touchpoints, axis=0)
        min_arg = np.amin(border_touchpoints, axis=0)
        self.eval_range = np.array([min_arg[self.chosen_features], max_arg[self.chosen_features]]).T

        result = np.array(border_touchpoints)
        num_per_point = int(num_samples / len(border_touchpoints))
        sigmas = (max_arg - min_arg) * sigma

        for point in border_touchpoints:
            mean = point
            cov = np.diag(sigmas ** 2)
            rand = np.random.multivariate_normal(mean, cov, num_per_point)
            result = np.append(result, rand, axis=0)

        return result

    @staticmethod
    def get_primary_features(data, labels, num_features):
        """ Returns most relevant *num_features* features using lars_path


        Args:
            data: the training data
            labels: labels for training / Y.
            num_features: Number of features desired

        Returns:
            used_features: list of indices of the relevant features in the data
        """
        _, _, coefs = lars_path(data,
                                labels,
                                method='lasso',
                                verbose=False)

        for i in range(len(coefs.T) - 1, 0, -1):
            nonzero = coefs.T[i].nonzero()[0]
            if len(nonzero) <= num_features:
                break
        used_features = nonzero

        return used_features

    def export_explainer(self):
        if isinstance(self.surrogate, DecisionTreeClassifier):
            export_tree(self.surrogate, file_name='tree_explainer.pdf')
        else:
            raise NotImplementedError('No visual or textual export supported for current explainer type')

    def get_surrogate(self):
        """

        Returns:

        """
        if self.surrogate is not None:
            return self.surrogate
        else:
            raise RuntimeError('No surrogate set yet, run train_surrogate() first')

    def train_surrogate(self, sample_set, features=None, num_features=None):
        pass

    def train_around_border(self, border_touchpoints, features=None, num_features=None):
        pass

class LinearSurrogate(SurrogateTrainer):

    def __init__(self, clf, chosen_features, alpha=0.1):
        super().__init__(clf, chosen_features)
        self.alpha = alpha

    def train_surrogate(self, sample_set, features=None, num_features=None):

        y = self.clf.predict(sample_set)

        used_features = []

        if features:
            used_features = features
        elif num_features and num_features > 0:
            # Find best num_features using lars
            used_features = SurrogateTrainer.get_primary_features(sample_set, y, num_features)
        else:
            used_features = range(sample_set.shape[1]) # Use all features available

        x = sample_set[:, used_features]

        self.train_ridge(x, y, self.alpha)
        return self.surrogate

    def train_around_border(self, border_touchpoints, features=None, num_features=None):
        sample_set = super().sample_normal(border_touchpoints, num_samples=500, sigma=0.1)
        return self.train_surrogate(sample_set, features, num_features)

    def train_ridge(self, x, y, alpha=0.1):
        """
        Trains a Ridge classifier on the sampled data and classifier predictions considering only
        the chosen_attributes for now, for simplicity
        """
        # TODO: Automate Parameters
        linear_clf = RidgeClassifier(alpha=alpha)
        linear_clf.fit(x, y)
        self.surrogate = linear_clf


class TreeSurrogate(SurrogateTrainer):

    def __init__(self, clf, chosen_features, max_depth=None):
        super().__init__(clf, chosen_features)
        self.max_depth = max_depth

    def train_surrogate(self, sample_set, features=None, num_features=None):

        y = self.clf.predict(sample_set)

        used_features = []

        if features:
            used_features = features
        elif num_features and num_features > 0:
            # Find best num_features using lars
            used_features = SurrogateTrainer.get_primary_features(sample_set, y, num_features)
        else:
            used_features = range(sample_set.shape[1]) # Use all features available

        x = sample_set[:, used_features]

        if self.max_depth is None:
            # Use number of features plus one as default
            self.max_depth = len(features) + 1

        self.train_tree(x, y, max_depth=self.max_depth)

        return self.surrogate

    def train_tree(self, x, y, max_depth=None):
        """

        Args:
            sample_set:
            num_features:
            max_depth:

        Returns:

        """

        if max_depth is None:
            max_depth = len(x[0]) + 1

        tree = DecisionTreeClassifier(max_depth=max_depth)

        tree.fit(x, y)
        self.surrogate = tree
