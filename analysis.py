import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class analysis():
    def __init__(self, m1, c1, m2, c2, m3, c3, attr1, attr2, mean, sigma, instance, X_test, Y_test, clf, eval_range):
        #--- Parameter für die Entscheidungsgrenzen
        self.lime_m = m1*sigma[attr2]/sigma[attr1]
        self.lime_c = mean[attr2] + c1*sigma[attr2] - m1*mean[attr1]*sigma[attr2]/sigma[attr1]
        self.own_m = m2
        self.own_c = c2
        self.ls_m = m3*sigma[attr2]/sigma[attr1]
        self.ls_c = mean[attr2] + c3*sigma[attr2] - m3*mean[attr1]*sigma[attr2]/sigma[attr1]
        self.instance = instance
        #--- Daten und trainiertes Modell
        self.x = X_test
        self.y = Y_test
        self.clf = clf
        #--- Evaluierungsbereich bestimmen
        #self.evaluation_cands = np.array(range(0,len(Y_test)))[clf.predict(X_test) == 0]
        #xxx = np.array([True,False,True,False,True,False,True,False])
        #yyy = np.array([False,True,False,True,False,True,False,True])
        xxx = []
        yyy = []
        for i in range(0,int(len(eval_range)/2)):
            xxx.append(eval_range[2*i])
            yyy.append(eval_range[2*i+1])
        xxx = np.array(xxx)
        yyy = np.array(yyy)
        
        self.eval_range = np.array([[np.min(xxx), np.max(xxx)], [np.min(yyy), np.max(yyy)]])
        #--- Referenzwerte (welche Seite von den Geraden ist als 0 klassifiziert)
        self.ref_lime = self.predict(self.lime_m, self.lime_c, [instance[attr1], instance[attr2]])
        self.ref_own = self.predict(self.own_m, self.own_c, [instance[attr1], instance[attr2]])
        self.ref_ls = self.predict(self.ls_m, self.ls_c, [instance[attr1], instance[attr2]])
    
    
    def drawAll(self, attr1, attr2):
        #--- Bestimme nun Punkte auf der LIME-Geraden im skalierten Raum
        x_line = np.linspace(self.eval_range[0][0], self.eval_range[0][1], 100)
        #y_line_lime = self.lime_m * x_line + self.lime_c
        
        #--- Reskalierung der Gerade
        #x_line = x_line * self.sigma[attr1] + self.mean[attr1]
        y_line_lime = self.lime_m * x_line + self.lime_c
        y_line_new = self.own_m * x_line + self.own_c
        y_line_ls = self.ls_m * x_line + self.ls_c
        
        #--- Zeichne LIME-Gerade
        plt.plot(x_line, y_line_lime, 'm-', lw=2)
        plt.plot(x_line, y_line_new, 'c-', lw=2)
        plt.plot(x_line, y_line_ls, 'g-', lw=2)
        
        #------
        #--- Visualize Attributspace
        sample = np.array(pd.read_csv("attribut_space.csv"))
        XX = np.array(sample)[:,1:3]
        yy = np.array(sample)[:,3]
        for i in range(0,len(yy)):
            yy[i] = int(yy[i] >= 0.5)
        yy = np.array(yy)
        
        plt.scatter(XX[:, 0], XX[:, 1], c=['limegreen'*int(yy[i])+'tomato'*(1-int(yy[i])) for i in range(0,len(yy))], cmap = plt.cm.Paired, marker='.', s=25)
        plt.scatter([self.instance[attr1]], [self.instance[attr2]], s=100, c='blue', marker='X')
        plt.xlabel("Attribut " + str(attr1))
        plt.ylabel("Attribut " + str(attr2))
        plt.title("Attributraum und dessen Klassifizierung")
        plt.show()
    
    
    def predict(self, m, c, punkt):
        #--- Schnittpunkt zwischen Gerade y=mx+c und der Gerade zwischen punkt und dem Ursprung
        referencePoint = [0, c]
        
        #--- Vektoren
        u1 = [punkt[0] - referencePoint[0], punkt[1] - referencePoint[1]]
        u2 = [1, m]
        
        #--- Vektoren normieren
        u1 = [u1[0]/np.sqrt(u1[0]*u1[0] + u1[1]*u1[1]), u1[1]/np.sqrt(u1[0]*u1[0] + u1[1]*u1[1])]
        u2 = [u2[0]/np.sqrt(u2[0]*u2[0] + u2[1]*u2[1]), u2[1]/np.sqrt(u2[0]*u2[0] + u2[1]*u2[1])]
        
        #--- Determinante berechnen
        det = u2[0]*u1[1] - u2[1]*u1[0]
        if(det < 0):
            return -1
        else:
            return 1
    
    
    def check(self, a, b):
        if(a == b):
            return 0
        else:
            return 1
    
    
    def evaluation(self, attr1, attr2, nsample=100):
        self.sample = []
        for i in range(0, nsample):
            coord1 = np.random.uniform(self.eval_range[0][0], self.eval_range[0][1])
            coord2 = np.random.uniform(self.eval_range[1][0], self.eval_range[1][1])
            dummy = copy.copy(self.instance)
            dummy[attr1] = coord1
            dummy[attr2] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
            coord4 = self.check(self.predict(self.lime_m, self.lime_c, [coord1, coord2]), self.ref_lime)
            coord5 = self.check(self.predict(self.own_m, self.own_c, [coord1, coord2]), self.ref_own)
            coord6 = self.check(self.predict(self.ls_m, self.ls_c, [coord1, coord2]), self.ref_ls)
            self.sample.append([i, coord1, coord2, coord3, coord4, coord5, coord6])
        
        #--- Visualize Attributspace
        yy = np.array(self.sample)[:,3]
        for i in range(0,len(yy)):
            self.sample[i][3] = int(yy[i] >= 0.5)
        plt.scatter(np.array(self.sample)[:,1], np.array(self.sample)[:,2], c=['green'*int(self.sample[i][3])+'red'*(1-int(self.sample[i][3])) for i in range(0,len(self.sample))], cmap = plt.cm.Paired, marker='.', s=25)
        plt.scatter([self.instance[attr1]], [self.instance[attr2]], s=100, c='blue', marker='X')
        
        
        x_line = np.linspace(self.eval_range[0][0], self.eval_range[0][1], 100)
        y_line_lime = self.lime_m * x_line + self.lime_c
        y_line_own = self.own_m * x_line + self.own_c
        y_line_ls = self.ls_m * x_line + self.ls_c
        plt.plot(x_line, y_line_lime, 'm-', lw=2)
        plt.plot(x_line, y_line_own, 'c-', lw=2)
        plt.plot(x_line, y_line_ls, 'g-', lw=2)
        plt.xlabel("Attribut " + str(attr1))
        plt.ylabel("Attribut " + str(attr2))
        plt.title("LIME (magenta); gg (cyan); ls (green)")
        plt.show()
        
        print(" ")
        print("Accuracy:")
        print("LIME:   ", np.sum(np.array(self.sample)[:,3] == np.array(self.sample)[:,4]), " (", nsample, ")")
        print("GG:     ", np.sum(np.array(self.sample)[:,3] == np.array(self.sample)[:,5]), " (", nsample, ")")
        print("LS:     ", np.sum(np.array(self.sample)[:,3] == np.array(self.sample)[:,6]), " (", nsample, ")")
        self.accuracies = [np.sum(np.array(self.sample)[:,3] == np.array(self.sample)[:,4])/nsample, np.sum(np.array(self.sample)[:,3] == np.array(self.sample)[:,5])/nsample, np.sum(np.array(self.sample)[:,3] == np.array(self.sample)[:,6])/nsample]
    
