import pandas as pd
import numpy as np
import sklearn

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_breast_cancer

KDD_COL_NAMES = np.array(["duration", "protocol_type", "service", "flag", "src_bytes",
                          "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                          "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                          "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                          "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                          "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                          "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                          "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                          "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                          "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels"])

IDS_DDOS_COL_NAMES_string = "Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min, Label"
IDS_DDOS_COL_NAMES_string = IDS_DDOS_COL_NAMES_string.replace(" ", "")
IDS_DDOS_COL_NAMES_string = IDS_DDOS_COL_NAMES_string.replace("\n", "")
IDS_DDOS_COL_NAMES_string = IDS_DDOS_COL_NAMES_string.lower()

IDS_DDOS_COL_NAMES = IDS_DDOS_COL_NAMES_string.split(',')

HELOC_STRING = "RiskPerformance,ExternalRiskEstimate,MSinceOldestTradeOpen,MSinceMostRecentTradeOpen,AverageMInFile,NumSatisfactoryTrades,NumTrades60Ever2DerogPubRec,NumTrades90Ever2DerogPubRec,PercentTradesNeverDelq,MSinceMostRecentDelq,MaxDelq2PublicRecLast12M,MaxDelqEver,NumTotalTrades,NumTradesOpeninLast12M,PercentInstallTrades,MSinceMostRecentInqexcl7days,NumInqLast6M,NumInqLast6Mexcl7days,NetFractionRevolvingBurden,NetFractionInstallBurden,NumRevolvingTradesWBalance,NumInstallTradesWBalance,NumBank2NatlTradesWHighUtilization,PercentTradesWBalance"
HELOC_NAMES = HELOC_STRING.split(',')

UCI_NAMES= np.array(["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"])


def load_heloc(normalize=False):

    # TODO: Remove features that have zero variance over the dataset (one value)
    # TODO: Remove features with very strong correlation
    data = pd.read_csv("data/heloc.csv")

    mapping = {'Good': 0, 'Bad': 1}
    data['RiskPerformance'] = data['RiskPerformance'].map(mapping)

    Y = np.array(data.pop("RiskPerformance"))
    data = data.astype('float64')
    X = np.array(data)
    if normalize:
        X = StandardScaler().fit_transform(X)

    return X, Y, HELOC_NAMES.remove('RiskPerfomance')

#--- Credit Daten aus CSV-Datei lesen
def load_data_txt(normalize=False):
    #--- Load data from txt-file
    data = pd.read_csv("data/UCI_Credit_Card.csv")

    Y = np.array(data.pop("default.payment.next.month"))
    data.pop("ID")
    X = np.array(data)
    if normalize:
        X = StandardScaler().fit_transform(X)

    return X, Y, UCI_NAMES

def load_kdd_csv(normalize=False, train=True):
    """
    Load NLS KDD dataset from csv
     - transform labels to binary (normal vs. attack) and omit categorical features for simplicity

    :param normalize: whether to normalize the feature values
    :param train: whether to get train dataset (True) or test (False)
    :return: The data and labels
    """

    categorical_columns = ["protocol_type", "service", "flag"]

    if train:
        data = pd.read_csv("data/kdd.csv", names=KDD_COL_NAMES)
    else:
        data = pd.read_csv("data/kdd_t.csv", names=KDD_COL_NAMES)

    DOS_LABELS = ['back','land','neptune', 'pod', 'smurt', 'teardrop', 'normal']
    data = data[ data['labels'].map(lambda  x: x in DOS_LABELS)]

    data['labels'] = np.where(data['labels'].str.match('normal'), 1, 0)
    data.drop(categorical_columns, axis=1, inplace=True)

    Y = data.pop('labels')
    X = data

    if normalize:
        X = StandardScaler().fit_transform(X)

    names = [e for e in KDD_COL_NAMES if e not in categorical_columns]
    names.remove('labels')
    return X, Y, names

def load_ids_csv(normalize=False, train=True):
    """

    :param normalize:
    :param train:
    :return: data, labels, feature names
    """

    data = pd.read_csv("data/ids/ddos.csv", names=IDS_DDOS_COL_NAMES)
    data['flowbytes/s'] = data['flowbytes/s'].astype('float64')
    data['flowpackets/s'] = data['flowpackets/s'].astype('float64')
    data['label'] = np.where(data['label'].str.match('BENIGN'), 1, 0) # train attacks as 0
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    data = data.loc[:, data.var() != 0] # remove columns with 0 variance

    Y = np.array(data.pop('label'))
    X = np.array(data)

    feature_names = IDS_DDOS_COL_NAMES
    feature_names.remove("label")

    if normalize:
        X = StandardScaler().fit_transform(X)

    return X, Y, feature_names


def load_data_iris():
    """
    Dataset is adapted to display a binary decision between versicolor and not-versicolor (or class 2 or not-class-2)
    Returns:

    """
    X, Y = load_iris(return_X_y=True)
    for i in range(0, len(Y)):
        if Y[i] == 1:
            Y[i] = 0
    for i in range(0, len(Y)):
        if Y[i] == 2:
            Y[i] = 1

    return X, Y


def load_data_breast_cancer():
    X, Y = load_breast_cancer(return_X_y=True)

    return X, Y


def load_data_survival():
    url = "data/haberman.csv"
    names = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']
    data = pd.read_csv(url, names=names)

    Y = np.array(data.pop('Survival status'))
    X = np.array(data)
    for i in range(0, len(Y)):
        if Y[i] == 1:
            Y[i] = 0
    for i in range(0, len(Y)):
        if Y[i] == 2:
            Y[i] = 1

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
