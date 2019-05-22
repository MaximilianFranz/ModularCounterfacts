import numpy as np
from utils import create_ranges, adjust_features, sample_normal, get_primary_features
from graph_export import export_tree

import sklearn
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from treeinterpreter import treeinterpreter as ti
from statsmodels import robust

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# noinspection PyPackageRequirements
from matplotlib import style
style.use("ggplot")

class ExplanationVisualizer():

    def __init__(self, explainer, features_to_show=None, feature_names=None, max_distance=0.3):
        self.explainer = explainer
        self.feature_names=feature_names

        # scoring for later acces
        self.tree_score = 0
        self.linear_score = 0

        self.tree_surrogate = None
        self.linear_surrogate = None
        self.max_distance = max_distance

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
            print('\n --------------------------------------')
            # self.explain_relative(self.explainer.last_instance, self.explainer.counterfactual, self.features_to_show)
            self.present_tolerance(self.explainer.last_instance, self.features_to_show)
            self.compare_surrogate()
            self.compare_lime()
            self.tree_accuracy_global()
            self.export_decision_tree()
            self.feature_importance()
            self.lars_features_local()
            self.feature_recall()



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

        self.plot_differences(instance, counterfactual, features)

        print('differences:', distance)

    def plot_differences(self, instance, counterfactual, selected_features):
        distance = (instance - counterfactual)
        eps = 1.0e-8
        distance[abs(distance) < eps] = 0

        ind = np.arange(len(selected_features))  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, instance[selected_features], width, alpha=0.7)
        p2 = plt.bar(ind, counterfactual[selected_features], width*0.5)

        plt.ylabel('normalized feature values')
        plt.title('Relative Feature Difference')
        plt.xticks(ind, np.array(self.feature_names)[selected_features])
        min = np.min(counterfactual[selected_features])
        max = np.max(counterfactual[selected_features])
        plt.yticks(np.linspace(min, max, num=10))
        plt.legend((p1[0], p2[0]), ('Instance', 'Counterfactual'))

        plt.show()

    def present_tolerance(self, instance, features):

        distances = np.sum(np.abs(instance[features] - self.explainer.touchpoints[:, features]), axis=1)


        self.confidence = np.min(distances[distances != 0])
        print('confidence score: ', self.confidence)
        self.prediction_proba = self.explainer.clf.predict_proba(self.explainer.last_instance.reshape(1, -1))
        print('prediction proba: ', self.prediction_proba)

    def construct_test_data_around_instance(self, instance, max_distance=0.3):
        """
        Sampling instances from the original dataset that are close to the instance given
        :param instance: Around which to sample
        :param max_distance: distance limit within which to sample
        :return:
        """
        dataset = self.explainer.dataset
        data_subset = dataset[np.random.randint(dataset.shape[0], size=20000), :]
        dist = np.sum(np.abs(data_subset - instance), axis=1) / data_subset.shape[1]
        result = data_subset[dist < max_distance]
        return result

    def compare_surrogate(self):

        # data = sample_normal(self.explainer.touchpoints, 500, 2)

        # Compare around decision boundary
        data = self.construct_test_data_around_instance(self.explainer.touchpoints[0], max_distance=self.max_distance)
        clf_pred = self.explainer.clf.predict(data)
        srg_pred = self.explainer.sg.surrogate.predict(data)


        self.linear_surrogate = self.explainer.sg.surrogate
        self.linear_score_db = accuracy_score(srg_pred, clf_pred)

        # Compare around original distance
        data = self.construct_test_data_around_instance(self.explainer.last_instance, max_distance=self.max_distance)
        clf_pred = self.explainer.clf.predict(data)
        srg_pred = self.explainer.sg.surrogate.predict(data)
        self.linear_score_instance = accuracy_score(srg_pred, clf_pred)
        print('accuracy surrogate around DB', self.linear_score_db)
        print('accuracy surrogate around instance', self.linear_score_instance)
        print('----------------------------- \n')

    def compare_lime(self):
        lime = self.create_lime_surrogate(self.explainer.last_instance, self.explainer.dataset, self.explainer.clf)

        data = self.construct_test_data_around_instance(self.explainer.touchpoints[0])
        clf_pred = self.explainer.clf.predict(data)
        srg_pred = lime.predict(data)

        self.lime_score_db = accuracy_score(srg_pred, clf_pred)

        data = self.construct_test_data_around_instance(self.explainer.last_instance, max_distance=self.max_distance)
        clf_pred = self.explainer.clf.predict(data)
        srg_pred = lime.predict(data)
        self.lime_score_instance = accuracy_score(srg_pred, clf_pred)

        print('LIME surrogate around DB', self.lime_score_db)
        print('LIME surrogate around instance', self.lime_score_instance)
        print('----------------------------- \n')


    def create_lime_surrogate(self, instance, train, clf):

        kernel_width = np.sqrt(train.shape[1]) * .75
        kernel_width = float(kernel_width)

        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        xss = self.explainer.dataset[np.random.randint(self.explainer.dataset.shape[0], size=7000)]
        yss = clf.predict(xss)

        distances = sklearn.metrics.pairwise_distances(
            xss,
            instance.reshape(1, -1),
            metric='cosine'
        ).ravel()


        clf = SGDClassifier()
        clf.fit(xss, yss, sample_weight=kernel(distances, kernel_width=kernel_width))

        return clf

    def export_decision_tree(self):

        data = self.construct_test_data_around_instance(self.explainer.touchpoints[0], max_distance=self.max_distance)
        clf_pred = self.explainer.clf.predict(data)
        X_train, X_test, Y_train, Y_test = train_test_split(data, clf_pred, test_size=0.2, random_state=1000)

        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X_train, Y_train)

        self.surrogate_features = np.array(self.feature_names)[np.flip(np.argsort(np.abs(tree.feature_importances_)))][0:10]

        data_db = self.construct_test_data_around_instance(self.explainer.touchpoints[0], max_distance=self.max_distance)
        tree_pred = tree.predict(X_test)
        clf_pred = self.explainer.clf.predict(X_test)

        export_tree(tree, 'exports/db_tree.pdf', self.feature_names)
        self.tree_surrogate = tree
        self.tree_score_db = accuracy_score(tree_pred, clf_pred)
        print('accuracy tree around DB', self.tree_score_db)

        data = self.construct_test_data_around_instance(self.explainer.last_instance, max_distance=self.max_distance)
        tree_pred = tree.predict(data)
        clf_pred = self.explainer.clf.predict(data)
        self.tree_score_instance = accuracy_score(tree_pred, clf_pred)
        print('accuracy tree around instance', self.tree_score_instance)
        print('----------------------------- \n')

    def tree_accuracy_global(self):

        data_subset = self.explainer.dataset[np.random.randint(self.explainer.dataset.shape[0], size=7000)]
        clf_pred = self.explainer.clf.predict(data_subset)

        linear = RidgeClassifier(alpha=1.0)
        linear.fit(data_subset, clf_pred)

        tree = DecisionTreeClassifier(max_depth=4, max_features=10)
        tree.fit(data_subset, clf_pred)

        selection = np.random.randint(self.explainer.testset.shape[0], size=2000)
        data_subset = self.explainer.testset[selection]
        clf_pred = self.explainer.clf.predict(data_subset)
        tree_pred = tree.predict(data_subset)
        linear_pred = linear.predict(data_subset)

        self.tree_global_features = np.array(self.feature_names)[np.flip(np.argsort(np.abs(tree.feature_importances_)))][0:10]

        self.tree_score_global = accuracy_score(tree_pred, self.explainer.testlabels[selection])
        self.linear_score_global = accuracy_score(linear_pred, self.explainer.testlabels[selection])
        print('GLOBAL tree accuracy (ccompared to ground truth): ', self.tree_score_global)
        print('GLOBAL linear accuracy (ccompared to ground truth): ', self.linear_score_global)
        print('GLOBAL tree feature importance ',
              list(zip(np.array(self.feature_names)[np.flip(np.argsort(np.abs(tree.feature_importances_))[-10:])],
                  np.flip(np.sort(np.abs(tree.feature_importances_))[-10:])))
              )

        print('GLOBAL linear feature importance ',
              list(zip(np.array(self.feature_names)[np.flip(np.argsort(np.abs(linear.coef_[0]))[-10:])],
              np.flip(np.sort(np.abs(linear.coef_[0]))[-10:])))
              )


    def feature_importance(self):
        """
        Trains a local surrogate random forest and returns its feature importance
        :return: feature importance of surrogate random forest
        """

        data_subset = self.construct_test_data_around_instance(self.explainer.touchpoints[0], max_distance=0.6)
        pred = self.explainer.clf.predict(data_subset)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(data_subset, pred)

        p, b, c = ti.predict(rf, self.explainer.last_instance.reshape(1, -1))
        c = c[0]
        print('FEATURE IMPORTANCES RF around DB: \n')

        for c, feature in sorted(zip(c[:,0],
                                     self.feature_names),
                                 key=lambda x: -abs(x[0]))[0:10]:
            print(feature, c)
        print('------------------------- \n')

    def lars_features_local(self):
        data_subset = self.construct_test_data_around_instance(self.explainer.last_instance, max_distance=0.6)
        labels = self.explainer.clf.predict(data_subset)
        features = get_primary_features(data_subset, labels, num_features=self.explainer.num_features)
        print('FEATURE IMPORTANCE LARS locally around instance')
        print(np.array(self.feature_names)[features])

    def feature_recall(self):
        """
        Comparing the gold-standard features of global tree with locally recalled features of the surrogate
        :return:
        """
        global_set = set(self.tree_global_features)
        local_set = set(self.surrogate_features)
        cut = [x for x in global_set if x in local_set]
        self.feature_recall_count = len(cut)
        print('Recall of global gold features:')
        print(len(cut), ' of 10')


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
            optimize = value + manhattan_distance(x)
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

