import numpy as np
from utils import create_ranges, adjust_features, sample_normal
from graph_export import export_tree

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from treeinterpreter import treeinterpreter as ti
from statsmodels import robust

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# noinspection PyPackageRequirements
from matplotlib import style
style.use("ggplot")

class ExplanationVisualizer():

    def __init__(self, explainer, features_to_show=None, feature_names=None):
        self.explainer = explainer
        self.feature_names=feature_names

        if not features_to_show:
            self.features_to_show = self.explainer.chosen_features
        else:
            self.features_to_show = features_to_show

        if self.explainer.sg.eval_range is not None:
            self.eval_range = self.explainer.sg.eval_range

    def present_explanation(self, method="visual", **kwargs):
        """
        Visual presentation of the surrogate

        TODO:
         - Choose most significant features by yourself in case of visual representation when given features
           are either None or greater than 2
         - Automatically provide textual explanation when to many features are given ..
         - Provide explanations for positive predictions (by how much was I accepted?)
         - counterfactual relative change as explanation

        Args:
            explainer:

        Returns:

        """
        if method == "visual":
            self.distance_heatmap(self.explainer.last_instance)
            # self.plot_results(self.explainer.last_instance, self.features_to_show)
        elif method == "relative":
            self.explain_relative(self.explainer.last_instance, self.explainer.counterfactual, self.features_to_show)
            self.present_tolerance(self.explainer.last_instance, self.features_to_show)
            self.compare_surrogate()
            self.export_decision_tree()
            self.feature_importance()



    def plot_results(self, normal_instance, features):
        """
        Plots results of the explanation if all steps have been performed

        Returns: -
        """
        # Works for 2D data only --> len(features) = 2
        if self.explainer.sg.eval_range is not None and len(features) > 2:
            use_features = self.explainer.chosen_features[0:2]
            print(use_features)
        elif len(features) == 2:
            print('use given features')
            use_features = features
        else:
            return

        # Limit number of features in eval range to allow meshgrid to work (max. 32 sequence length)
        if self.eval_range[0].shape[0] > 32:
            eval_data = create_ranges(self.eval_range[0][use_features], self.eval_range[1][use_features], 5)
        else:
            eval_data = create_ranges(self.eval_range[0], self.eval_range[1], 5)


        # grid_data = np.array(np.meshgrid(*eval_data, sparse=False, indexing='ij')).reshape(len(use_features), -1).T
        # eval_data_full = adjust_features(normal_instance, use_features, grid_data, 0)
        #
        # exp_pred = self.explainer.clf.predict(eval_data_full)

        cov = np.diag(np.full(len(use_features), 0.02))
        rand = np.random.multivariate_normal(self.explainer.counterfactual[use_features], cov, 200)
        rand = adjust_features(normal_instance, use_features, rand, 0)

        rand_pred = self.explainer.clf.predict(rand)

        color_map = ['tomato', 'limegreen']

        plt.scatter(*self.explainer.counterfactual[use_features], c='r', marker="X", s=100)
        plt.scatter(*normal_instance[use_features], c='b', marker="X", s=100)
        plt.scatter(self.explainer.touchpoints[:, use_features[0]], self.explainer.touchpoints[:, use_features[1]], c='purple', s=20, marker=".")
        # plt.scatter(eval_data_full[:, use_features[0]], eval_data_full[:, use_features[1]], c=exp_pred, cmap=ListedColormap(color_map), s=10, marker=".")
        plt.scatter(rand[:, use_features[0]], rand[:, use_features[1]], c=rand_pred, cmap=ListedColormap(color_map), s=10, marker=".")
        plt.show()

    def explain_relative(self, instance, counterfactual, features):

        print('your features: ', instance[features])
        print('desired features: ', counterfactual[features])
        distance = (instance - counterfactual)[features]
        eps = 1.0e-8
        distance[abs(distance) < eps] = 0
        print('differences:', distance)

    def present_tolerance(self, instance, features):

        distances = np.abs(instance[features] - self.explainer.touchpoints[:, features])
        min = np.argmin(distances, axis=0)

        distance = instance[features] - self.explainer.touchpoints[min, features]
        eps = 1.0e-8
        distance[abs(distance) < eps] = 0
        print('minimum required change: ', distance)

    def compare_surrogate(self):

        data = sample_normal(self.explainer.touchpoints, 500, 2)
        clf_pred = self.explainer.clf.predict(data)
        srg_pred = self.explainer.sg.surrogate.predict(data)

        print('accuracy surrogate ', accuracy_score(srg_pred, clf_pred))

    def export_decision_tree(self):

        data = sample_normal(self.explainer.touchpoints, 500, 2)
        clf_pred = self.explainer.clf.predict(data)

        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(data, clf_pred)

        tree_pred = tree.predict(data)

        export_tree(tree, 'exports/db_tree.pdf', self.feature_names)
        print('accuracy tree ', accuracy_score(tree_pred, clf_pred))

    def feature_importance(self):
        """
        Trains a global surrogate random forest and returns its feature importance
        :return: feature importance of surrogate random forest
        """

        data_subset = self.explainer.dataset[np.random.randint(self.explainer.dataset.shape[0], size=7000), :]
        pred = self.explainer.clf.predict(data_subset)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(data_subset, pred)

        p, b, c = ti.predict(rf, self.explainer.last_instance.reshape(1, -1))
        c = c[0]

        for c, feature in sorted(zip(c[:,0],
                                     self.feature_names),
                                 key=lambda x: -abs(x[0]))[0:10]:
            print(feature, c)

    def distance_heatmap(self, instance, fixed_features=True):

        mad = np.array(robust.mad(self.explainer.dataset, axis=0))
        non_zero = mad[mad != 0] # make sure not to devide by zero

        x_feature, y_feature = 0,0

        # find features with both variance in the and significant effect
        for feature in self.explainer.chosen_features:
            if mad[feature] != 0 and x_feature == 0:
                x_feature = feature
            elif mad[feature] != 0 and y_feature == 0:
                y_feature = feature

        delta = 1

        # Use first / most significant features instead of the selected ones
        if fixed_features:
            x_feature, y_feature = self.explainer.chosen_features[0:2]

        x_range = [instance[x_feature] - delta, instance[x_feature] + delta]
        y_range = [instance[y_feature] - delta, instance[y_feature] + delta]


        # Generate instance grid on two dimensions
        xs, ys = np.linspace(*x_range, 10), np.linspace(*y_range, 10)
        XS, YS = np.meshgrid(xs, ys)
        XS, YS = XS.flatten(), YS.flatten()
        updates = np.array(list(zip(XS, YS)))


        results_combined = np.empty((10, 10))
        results_prediction = np.empty((10, 10))
        results_metric = np.empty((10, 10))

        mad = np.array(robust.mad(self.explainer.dataset, axis=0))*10
        non_zero = mad[mad != 0] # make sure not to devide by zero

        # Copied distance metric from counterfactual
        # TODO: Refactor functions to avoid Copy
        def manhattan_distance(y, x=instance, weigths=None):

            if weigths is None:
                weigths = np.full(len(non_zero), 1)

            abs = np.abs(x - y)[mad != 0]

            result = np.nansum(np.divide(abs, non_zero) * weigths)
            if np.isinf(result):
                return 0
            return result / len(non_zero)

        def func(x, l=10):

            value = (1 - self.explainer.clf.predict_proba(x.reshape(1, -1))[0, 1])
            optimize = value + manhattan_distance(x)/l
            return optimize


        all_instances = adjust_features(instance, [x_feature, y_feature], updates, 0)
        i = 0

        # combined
        for instance in all_instances:
            results_combined[i // 10][i % 10] = func(instance)
            i += 1

        i = 0
        for instance in all_instances:
            results_prediction[i // 10][i % 10] = (1 - self.explainer.clf.predict_proba(instance.reshape(1, -1))[0, 1])
            i += 1

        i = 0
        for instance in all_instances:
            results_metric[i // 10][i % 10] = manhattan_distance(instance)
            i += 1

        fig, (s1, s2, s3) = plt.subplots(1,3, sharex=True, sharey=True)
        # s1 = fig.add_subplot(1, 1, 1, xlabel=self.feature_names[x_feature], ylabel=self.feature_names[y_feature])
        s1.title.set_text('Combined')
        im = s1.imshow(results_combined, cmap=plt.cm.RdBu, extent=(x_range[0],x_range[1], y_range[0], y_range[1]), interpolation='bilinear')
        plt.colorbar(im, ax=s1)

        # s2 = fig.add_subplot(1, 1, 1, xlabel=self.feature_names[x_feature], ylabel=self.feature_names[y_feature])
        s2.title.set_text('Metric')
        im = s2.imshow(results_metric, cmap=plt.cm.RdBu, extent=(x_range[0],x_range[1], y_range[0], y_range[1]), interpolation='bilinear')
        plt.colorbar(im, ax=s2)

        # s3 = fig.add_subplot(1, 1, 1, xlabel=self.feature_names[x_feature], ylabel=self.feature_names[y_feature])
        s3.title.set_text('Prediction')
        im = s3.imshow(results_prediction, cmap=plt.cm.RdBu, extent=(x_range[0],x_range[1], y_range[0], y_range[1]), interpolation='bilinear')
        plt.colorbar(im, ax=s3)

        plt.savefig('exports/heatmap.png')
        plt.show()

