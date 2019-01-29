import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import Ridge, RidgeClassifier

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class limeExplainer():
    def __init__(self, dataset, instance, attr, clf):
        self.attr = attr
        self.instance = instance
        self.clf = clf
        self.dataset = np.array(dataset)

        self.mean = np.sum(dataset, axis=0)/len(dataset)
        self.sigma = np.sqrt(np.sum((dataset-self.mean)*(dataset-self.mean), axis=0)/len(dataset))

        self.attr_range = []
        for item in attr:
            self.attr_range.append([min(dataset[:,item]), max(dataset[:,item])])

        self.instance_trafo = self.transform([instance[attr[0]], instance[attr[1]]])

    def get_explainer(self):
        return self.explainer

    def get_name(self):
        return 'lime'

    def locality(self, z0, local=1.0):
        res1 = np.exp(-(self.instance_trafo[0] - z0[0])*(self.instance_trafo[0] - z0[0])/(local*local))
        res2 = np.exp(-(self.instance_trafo[1] - z0[1])*(self.instance_trafo[1] - z0[1])/(local*local))
        return res1*res2


    def transform(self, z):
        """
        Normalizes inputs around mean with standard deviation

        Maps to normalized space
        """
        return [(z[0] - self.mean[self.attr[0]])/self.sigma[self.attr[0]], (z[1] - self.mean[self.attr[1]])/self.sigma[self.attr[1]]]

    def retransform(self, z):
        """
        Maps instance back into the unnormalized space
        """
        return [z[0] * self.sigma[self.attr[0]] + self.mean[self.attr[0]], z[1] * self.sigma[self.attr[1]] + self.mean[self.attr[1]]]


    def explain(self, nsample=500, local=1.0):
        Z = []
        self.Z_raw = []
        for i in range(0, nsample):
            #z = [np.random.normal(self.instance[self.attr[0]], 3*self.sigma[0]), np.random.normal(self.instance[self.attr[1]], 3*self.sigma[1])]
            z = np.array([np.random.uniform(self.attr_range[0][0], self.attr_range[0][1]), np.random.uniform(self.attr_range[1][0], self.attr_range[1][1])])
            dummy = np.array(self.instance)
            dummy[self.attr] = z
            # Add uniformly sampled points in the attribute range
            self.Z_raw.append([])
            self.Z_raw[i].append(z[0])
            self.Z_raw[i].append(z[1])
            Z.append([])
            # Normalize the sample to compute normalized distance / locality
            z0 = self.transform(z)
            Z[i].append(z0[0])
            Z[i].append(z0[1])
            # Classify the raw instance instance and add it to the sample
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
            Z[i].append(int(coord3 >= 0.5))
            # Add a locality feature, higher if close to the original instance
            # lower if further away. Computed in normalized space
            Z[i].append(self.locality(z0, local))
        self.Z = Z

        # Train Ridge classifier on weighted features
        clf = RidgeClassifier(alpha=1.0)
        X=np.array(Z)[:,0:2]
        y=np.array(Z)[:,2]
        w=np.array(Z)[:,3]
        clf.fit(X, y, w)
        self.explainer = clf
        # self.lime_m = -clf.coef_[0] / clf.coef_[1]
        # self.lime_c = (0.5 - clf.intercept_) / clf.coef_[1]

        # self.lime_m_trafo = self.lime_m*self.sigma[self.attr[1]]/self.sigma[self.attr[0]]
        # self.lime_c_trafo = self.mean[self.attr[1]] + self.lime_c*self.sigma[self.attr[1]] - self.lime_m*self.mean[self.attr[0]]*self.sigma[self.attr[1]]/self.sigma[self.attr[0]]


    def drawLIME(self, instance, attr1, attr2):
        #--- Bestimme nun Punkte auf der LIME-Geraden im skalierten Raum
        x_line = np.linspace(self.attr_range[0][0], self.attr_range[0][1], 100)
        y_line = self.lime_m_trafo * x_line + self.lime_c_trafo

        #--- Zeichne LIME-Gerade
        plt.plot(x_line, y_line, 'm-', lw=1)

        #--- Visualize Attributspace
        sample = np.array(pd.read_csv("attribut_space.csv"))
        XX = np.array(sample)[:,1:3]
        yy = np.array(sample)[:,3]
        for i in range(0,len(yy)):
            yy[i] = int(yy[i] >= 0.5)
        yy = np.array(yy)

        #plt.scatter(XX[:, 0], XX[:, 1], c=['limegreen'*int(yy[i])+'tomato'*(1-int(yy[i])) for i in range(0,len(yy))], cmap = plt.cm.Paired, marker='.', s=25)
        plt.scatter(np.array(self.Z_raw)[:, 0], np.array(self.Z_raw)[:, 1], c=['limegreen'*int(np.array(self.Z)[i, 2])+'tomato'*(1-int(np.array(self.Z)[i, 2])) for i in range(0,len(np.array(self.Z)[:, 2]))], cmap = plt.cm.Paired, marker='.', s=25)
        plt.scatter([instance[attr1]], [instance[attr2]], s=100, c='blue', marker='X')
        plt.xlabel("Attribut " + str(attr1))
        plt.ylabel("Attribut " + str(attr2))
        plt.title("Attributraum und dessen Klassifizierung")
        plt.show()
