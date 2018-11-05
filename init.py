import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


#--- Credit Daten aus CSV-Datei lesen
def load_data_txt():
    #--- Load data from txt-file
    data = pd.read_csv("UCI_Credit_Card.csv")
    
    Y = np.array(data.pop("default.payment.next.month"))
    data.pop("ID")
    X = np.array(data)
    
    return X, Y



#--- Create Attributespace
def createSpace(nsample, attr1, attr2, instance, X, clf):
    min1,max1 = np.min(X[:,attr1]),np.max(X[:,attr1])
    min2,max2 = np.min(X[:,attr2]),np.max(X[:,attr2])
    
    sample = []
    for j in range(0, nsample):
        coord1 = np.random.uniform(min1, max1)
        coord2 = np.random.uniform(min2, max2)
        dummy = list(instance)
        dummy[attr1] = coord1
        dummy[attr2] = coord2
        coord3 = np.array(clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
        sample.append([j, coord1, coord2, coord3])
    
    fd = open('attribut_space.csv', "w")
    for item in sample:
        fd.write(str(item[0]) + "," + str(item[1]) + "," + str(item[2]) + "," + str(item[3]) + "\n")
    fd.close()



#--- Show Attributespace
def showSpace(attr1, attr2, instance):
    sample = np.array(pd.read_csv("attribut_space.csv"))
    
    #--- Visualize Attributspace
    XX = np.array(sample)[:,1:3]
    yy = np.array(sample)[:,3]
    for i in range(0,len(yy)):
        yy[i] = int(yy[i] >= 0.5)
    yy = np.array(yy)
    
    plt.scatter(XX[:, 0], XX[:, 1], c=['green'*int(yy[i])+'red'*(1-int(yy[i])) for i in range(0,len(yy))], cmap = plt.cm.Paired, marker='.', s=25)
    plt.scatter([instance[attr1]], [instance[attr2]], s=100, c='blue', marker='X')
    plt.xlabel("Attribut " + str(attr1))
    plt.ylabel("Attribut " + str(attr2))
    plt.title("Attributraum und dessen Klassifizierung")
    plt.show()