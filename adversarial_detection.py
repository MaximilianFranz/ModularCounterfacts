from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import gradientgrow
import init
import time
from magnetic_sampling import MagneticSampler

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class AdversarialDetection():

    def __init__(self, X, clf, chosen_attributes=None):
        """
        X: the dataset
        clf: must be fit before
        """
        self.X = X
        self.clf = clf
        self.chosen_attributes = chosen_attributes
        self.scaler = StandardScaler().fit(self.X)
        self.ms = MagneticSampler(clf, None)



    def load_data_txt(self, normalize=False):
        data = pd.read_csv("UCI_Credit_Card.csv")

        Y = np.array(data.pop("default.payment.next.month"))
        data.pop("ID")
        X = np.array(data)
        if normalize:
            X = StandardScaler().fit_transform(X)

        return X, Y


    def get_border_touchpoints(self, support_points, original_instance, predictor_fn, fineness=5):
        """
        Uses a sort of binary search to find the border touchpoint on segments faster.
        """
        touchpoints = []
        for point in support_points:
            # search on segment between point and original_instance with binary search
            l = 0
            r = 1
            for i in range(fineness):
                m = (r + l)/ 2.0
                x_m = original_instance + m*(point - original_instance)
                if predictor_fn.predict_proba(np.array(x_m).reshape(1,-1))[0,1] <= 0.5:
                    l = m
                else:
                    r = m
            touchpoints.append(x_m)

        self.boundary_touchpoints = np.array(touchpoints)
        return self.boundary_touchpoints


    def sample_around(self, border_touchpoints, num_samples, sigma):
        """
        Samples around the border_touchpoints with a normal distribution to generate
        a dataset for training a linear model which yields the explanation

        Normal distribution is parametrized based on the distribution of
        the border_touchpoints, so that we sample along the decision boundary

        """
        max_arg = np.amax(border_touchpoints, axis=0)
        min_arg = np.amin(border_touchpoints, axis=0)
        result = np.array(border_touchpoints)
        num_per_point = int(num_samples / len(border_touchpoints))
        sigmas = (max_arg - min_arg)*sigma
        for point in border_touchpoints:
            mean = point
            cov = np.diag(sigmas**2)
            rand = np.random.multivariate_normal(mean, cov, num_per_point)
            result = np.append(result, rand, axis=0)

        self.sample_set = result
        return self.sample_set


    def get_first_adversarial(self, original_instance):
        """
        Using the GradientSearch approach this method searches the first adversarial
        instance to feed into magnetic_sampling.
        """
        # GradientSearch returns only the two chosen attributes
        dec = gradientgrow.Decision(self.X,
                                    self.chosen_attributes,
                                    original_instance,
                                    self.clf)
        dec.gradient_search(step=0.05, scale=1.0, nsample=100)
        self.gradientGrow = dec
        self.first_adversarial = dec.get_last_instance()
        return self.first_adversarial

    def train_explainer(self):
        """
        Trains a Ridge classifier on the sampled data considering only
        the chosen_attributes for now, for simplicity
        """
        X = self.sample_set[:, self.chosen_attributes]
        y = self.predictions
        clf = Ridge(alpha=0.1)
        clf.fit(X,y)
        self.explainer = clf
        return self.explainer

    def explain_instance(self, instance, num_samples=1000, locality=0.1, chosen_attributes=None):
        """
        Using the functions provided in this module this returns a linear
        """
        self.instance = instance
        if chosen_attributes is not None:
            self.chosen_attributes = chosen_attributes
        # TODO: Make sure to handle cases without chosen_attributes accordingly
        self.first_adversarial = self.get_first_adversarial(instance)

        # magnetic_sampling uses the predictor_fn not the predictor,
        # thus pass the corresponding fct
        one = time.time()
        self.support_points = self.ms.magnetic_sampling(self.clf.predict_proba,
                                instance,
                                self.first_adversarial,
                                num_samples=5,
                                features=self.chosen_attributes,
                                sector_width=0.35,
                                confidence=5,
                                threshold=2
                                )
        two = time.time()
        print('ms time: ', two - one)

        one = time.time()
        self.border_touchpoints = self.get_border_touchpoints(self.support_points,
                                                    self.instance,
                                                    self.clf,
                                                    fineness=5)
        two = time.time()
        print('border_touchpoints time: ', two - one)

        one = time.time()
        self.sample_set = self.sample_around(self.border_touchpoints, num_samples, locality)
        self.predictions = self.clf.predict_proba(self.sample_set)[:,1]
        two = time.time()
        print('sample-predict time: ', two - one)

        one = time.time()
        self.predictions = np.round(self.predictions)
        self.explainer = self.train_explainer()
        self.explainer_prediction = self.explainer.predict(self.sample_set[:, self.chosen_attributes])
        two = time.time()
        print('train explainer time: ', two - one)

        # Clf trained on only 2D data -> take the only two coefs
        self.m = (-1)*self.explainer.coef_[0] / self.explainer.coef_[1]
        self.b = (0.5 - self.explainer.intercept_) / self.explainer.coef_[1]

    def plot_results(self):
        """
        Plots results of the explanation if all steps have been performed

        Returns: -
        """
        if self.predictions is not None:

            pos = self.predictions[self.predictions > 0.5]
            neg = self.predictions[self.predictions <= 0.5]
            print('positive ', pos.size)
            print('negative ', neg.size)

            print('exp pos: ', self.explainer_prediction[self.explainer_prediction > 0.5].size)
            score = self.explainer.score(self.sample_set[:, self.chosen_attributes], self.predictions)
            print('score of explainer ', score )

            # generate colormap for predictions
            colors = []
            for y_i in self.predictions:
                if y_i > 0.5:
                    colors.append('limegreen')
                else:
                    colors.append('tomato')


            # For brevity
            attr1 = self.chosen_attributes[0]
            attr2 = self.chosen_attributes[1]

            # Create explainer space for 2-D representation
            x_min = np.min(self.sample_set[:, attr1])
            x_max = np.max(self.sample_set[:, attr1])
            x_line = np.linspace(x_min, x_max, 100)
            y_line = self.m * x_line + self.b

            # Plots
            plt.scatter(self.sample_set[:, attr1], self.sample_set[:, attr2], c=colors, cmap = plt.cm.Paired, marker='.', s=25)
            plt.scatter([self.instance[attr1]], [self.instance[attr2]], s=100, c='blue', marker='X')
            plt.scatter([self.first_adversarial[attr1]], [self.first_adversarial[attr2]], s=100, c='red', marker='X')
            plt.scatter([self.support_points[:, attr1]], [self.support_points[:, attr2]], s=30, c='purple', marker='o')
            plt.scatter([self.border_touchpoints[:, attr1]], [self.border_touchpoints[:, attr2]], s=100, c='black', marker='.')
            plt.plot(x_line, y_line, 'b-', lw=1)
            plt.xlabel("Attribut " + str(attr1))
            plt.ylabel("Attribut " + str(attr2))
            plt.title("Preliminary Results of adversarial Detection")
            plt.show()


def test():

    chosen_attributes = [0,5]
    clf = RandomForestClassifier(n_jobs=100, n_estimators=50, random_state=5000)
    X, Y = init.load_data_txt(normalize=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)
    clf.fit(X_train, Y_train)

    explainer = AdversarialDetection(X, clf=clf, chosen_attributes=chosen_attributes)

    explainer.explain_instance(X_test[15], num_samples=600)
    explainer.plot_results()

if __name__ == '__main__':
    test()
