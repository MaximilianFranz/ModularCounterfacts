import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class localSurrogate():
    def __init__(self, dataset, instance, attr, clf):
        self.instance = instance
        self.attr = attr
        self.clf = clf
        self.dataset = np.array(dataset)
        self.mean = np.sum(dataset, axis=0)/len(dataset)
        self.sigma = np.sqrt(np.sum((dataset-self.mean)*(dataset-self.mean), axis=0)/len(dataset))
        self.instance_trafo = self.transform([instance[attr[0]], instance[attr[1]]])
    
    
    def rand_sphere(self, a0, a1):
        dim = 2
        x = np.random.normal(0,1,dim)
        u = np.power(np.random.uniform(np.power(a0,dim),np.power(a1,dim)), (1/dim))
        x = u * x / np.sqrt(np.sum(x*x))
        return x
    
    
    def sampling(self, nsample, a0, a1):
        Z = []
        for i in range(0,nsample):
            z = self.rand_sphere(a0, a1)
            Z.append([])
            Z[i].append(i)
            Z[i].append(z[0])
            Z[i].append(z[1])
        
        inst_trafo = self.transform([self.instance[self.attr[0]], self.instance[self.attr[1]]])
        for i in range(0, nsample):
            coord1 = Z[i][1] + inst_trafo[0]
            coord2 = Z[i][2] + inst_trafo[1]
            z = self.retransform([coord1, coord2])
            Z[i][1] = z[0]
            Z[i][2] = z[1]
            dummy = copy.copy(self.instance)
            dummy[self.attr[0]] = z[0]
            dummy[self.attr[1]] = z[1]
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
            coord3 = int(coord3 >= 0.5)
            Z[i].append(coord3)
        Z = np.array(Z)
        plt.scatter(Z[:,1], Z[:,2], c=['limegreen'*int(Z[i,3])+'tomato'*(1-int(Z[i,3])) for i in range(0,len(Z[:,3]))], cmap = plt.cm.Paired, marker='.', s=25)
        plt.show()
        return Z
    
    
    def transform(self, z):
        return [(z[0] - self.mean[self.attr[0]])/self.sigma[self.attr[0]], (z[1] - self.mean[self.attr[1]])/self.sigma[self.attr[1]]]
    
    def retransform(self, z):
        return [z[0] * self.sigma[self.attr[0]] + self.mean[self.attr[0]], z[1] * self.sigma[self.attr[1]] + self.mean[self.attr[1]]]
    
    
    def growingSpheres(self, nsample=60, eta=1.0):
        check = 1
        while check > 0:
            Z = np.array(self.sampling(nsample=nsample, a0=0, a1=eta))
            check = np.sum(Z[:,3])
            if check > 0:
                eta = eta/2
        
        a0 = eta
        a1 = 2*eta
        check = 0
        while check < 1:
            Z = np.array(self.sampling(nsample=nsample, a0=a0, a1=a1))
            check = np.sum(Z[:,3])
            if check < 1:
                a0 = a1
                a1 = a1 + eta
        
        #choice = int(np.random.choice(np.array(Z[Z[:,3]==1])[:,0]))
        cands = np.array(Z[Z[:,3]==1])[:,0]
        inst_trafo = self.transform([self.instance[self.attr[0]], self.instance[self.attr[1]]])
        min_dist = 1000
        min_cand = 0
        for i in list(cands):
            z = [Z[int(i),1], Z[int(i),2]]
            z_trafo = self.transform(z)
            new_dist = np.sqrt((inst_trafo[0] - z_trafo[0])*(inst_trafo[0] - z_trafo[0]) + (inst_trafo[1] - z_trafo[1])*(inst_trafo[1] - z_trafo[1]))
            if min_dist > new_dist:
                min_dist = new_dist
                min_cand = int(i)
        choice = min_cand
        
        self.mittelpunkt = self.transform([Z[choice][1], Z[choice][2]])
        self.radius = a1 - a0
    
    
    def explain_ls(self, nsample=200):
        self.Z = []
        self.Z0 = []
        for i in range(0, nsample):
            rand = self.rand_sphere(0,self.radius/2)
            z = [self.mittelpunkt[0] + rand[0], self.mittelpunkt[1] + rand[1]]
            self.Z.append([])
            self.Z[i].append(z[0])
            self.Z[i].append(z[1])
            
            z0 = self.retransform(z)
            dummy = copy.copy(self.instance)
            dummy[self.attr[0]] = z0[0]
            dummy[self.attr[1]] = z0[1]
            self.Z0.append([])
            self.Z0[i].append(z0[0])
            self.Z0[i].append(z0[1])
            
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
            self.Z[i].append(int(coord3 >= 0.5))
        
        clf = Ridge(alpha=1.0)
        X=np.array(self.Z)[:,0:2]
        y=np.array(self.Z)[:,2]
        clf.fit(X, y)
        self.ls_m = -clf.coef_[0] / clf.coef_[1]
        self.ls_c = (0.5 - clf.intercept_) / clf.coef_[1]
    
    
    def drawLS(self, attr1, attr2):
        #--- Bestimme nun Punkte auf der LIME-Geraden im skalierten Raum
        x_line = np.linspace(self.mittelpunkt[0] - self.radius, self.mittelpunkt[0] + self.radius, 100)
        y_line = self.ls_m * x_line + self.ls_c
        
        #--- Punkte reskalieren
        x_line0 = []
        y_line0 = []
        for i in range(0, len(x_line)):
            z = self.retransform([x_line[i], y_line[i]])
            x_line0.append(z[0])
            y_line0.append(z[1])
        
        #--- Zeichne LIME-Gerade
        plt.plot(x_line0, y_line0, 'm-', lw=1)
        
        
        #------
        #--- Visualize Attributspace
        sample = np.array(pd.read_csv("attribut_space.csv"))
        yy = np.array(sample)[:,3]
        for i in range(0,len(yy)):
            yy[i] = int(yy[i] >= 0.5)
        yy = np.array(yy)
        
        plt.scatter(np.array(self.Z0)[:, 0], np.array(self.Z0)[:, 1], c=['limegreen'*int(np.array(self.Z)[i,2])+'tomato'*(1-int(np.array(self.Z)[i,2])) for i in range(0,len(np.array(self.Z)[:,2]))], cmap = plt.cm.Paired, marker='.', s=25)
        plt.scatter([self.instance[attr1]], [self.instance[attr2]], s=100, c='blue', marker='X')
        plt.scatter(self.retransform(self.mittelpunkt)[0], self.retransform(self.mittelpunkt)[1], s=100, c='blue', marker='+')
        plt.xlabel("Attribut " + str(attr1))
        plt.ylabel("Attribut " + str(attr2))
        plt.title("Attributraum und dessen Klassifizierung")
        plt.show()
