from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

from explainer import DefaultExplainer
from visualizer import ExplanationVisualizer
from data_loader import load_data_txt, load_heloc, load_kdd_csv, load_ids_csv

import utils
import numpy as np

def test():
    chosen_attributes = [0, 5]
    clf = RandomForestClassifier(n_jobs=100, n_estimators=100, random_state=5000)
    X, Y, names = load_data_txt(normalize=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1001)
    clf.fit(X_train, Y_train)

    explainer = DefaultExplainer(clf, X, [0, 5])

    for instance in X_test:
        if clf.predict(instance.reshape(1, -1)) < 0.5:
            explainer.explain_instance(instance)
            break

    y_clf = clf.predict(X_test)
    y_exp = explainer.sg.surrogate.predict(X_test)
    print('comparison score on test dataset: ', accuracy_score(y_clf, y_exp))

    viz = ExplanationVisualizer(explainer, chosen_attributes, feature_names=names)
    viz.present_explanation('relative')

def test_heloc():
    clf = RandomForestClassifier(n_jobs=100, n_estimators=100, random_state=1000)
    X, Y, names = load_heloc(normalize=False)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1200)
    clf.fit(X_train, Y_train)


    print('accuracy Rf: ', accuracy_score(clf.predict(X_test), Y_test))

    explainer = DefaultExplainer(clf, X)

    for instance in X_test:
        if clf.predict_proba(instance.reshape(1, -1))[0, 1]  < 0.5:
            explainer.explain_instance(instance)
            break

    viz = ExplanationVisualizer(explainer, feature_names=names)
    viz.present_explanation(method='relative')

def runon_kdd():

    mlp_dump_file = "exports/mlp.joblib"
    p = Path(mlp_dump_file)

    X, Y, names= load_kdd_csv(normalize=True, train=True)

    if p.is_file():
        clf = joblib.load(mlp_dump_file)
    else:
        clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes = (20, 5), random_state = 1)
        clf.fit(X, Y)
        joblib.dump(clf, mlp_dump_file)

    Xtest, Ytest, names = load_kdd_csv(normalize=True, train=False)
    print('accuracy MLP: ', accuracy_score(clf.predict(Xtest), Ytest))

    explainer = DefaultExplainer(clf, X, None)

    for instance in Xtest:
        if clf.predict(instance.reshape(1, -1)) < 0.5:
            explainer.explain_instance(instance)
            break

    print('counterfact: ', clf.predict(explainer.counterfactual.reshape(1, -1)))
    viz = ExplanationVisualizer(explainer, None, feature_names=names)
    viz.present_explanation(method='relative')

def runon_ids():

    mlp_dump_file = "exports/mlp_ids.joblib"
    p = Path(mlp_dump_file)

    X, Y, names = load_ids_csv(normalize=True)

    choose = np.random.randint(X.shape[0], size=50)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1200)

    if p.is_file():
        clf = joblib.load(mlp_dump_file)
    else:
        clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes = (20, 5), random_state = 1)
        clf.fit(X_train, Y_train)
        joblib.dump(clf, mlp_dump_file)

    explainer = DefaultExplainer(clf, X, None)

    for instance in X_test:
        if clf.predict(instance.reshape(1, -1)) < 0.5: # explain attacks (0)
            explainer.explain_instance(instance)
            break

    print('accuracy MLP: ', accuracy_score(clf.predict(X_test), Y_test))

    viz = ExplanationVisualizer(explainer, None, feature_names=names)
    viz.present_explanation(method='visual')

def tsne_on_ids():

    mlp_dump_file = "exports/mlp_ids.joblib"
    p = Path(mlp_dump_file)

    X, Y, names = load_ids_csv(normalize=True)

    choose = np.random.randint(X.shape[0], size=50)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1201)

    if p.is_file():
        clf = joblib.load(mlp_dump_file)
    else:
        clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes = (20, 5), random_state = 1)
        clf.fit(X_train, Y_train)
        joblib.dump(clf, mlp_dump_file)

    data_subset = X[choose, :]
    label_subset = Y[choose]
    explainer = DefaultExplainer(clf, X, None)
    counterfacts = explainer.get_counterfactuals(data_subset[label_subset == 0])

    distances = abs(counterfacts - data_subset[label_subset == 0])
    print(np.average(distances, axis=0))

    utils.plot_tsne(data_subset, label_subset, counterfacts)

    return

if __name__ == '__main__':
    tsne_on_ids()
