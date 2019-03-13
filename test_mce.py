from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

from explainer import DefaultExplainer
from visualizer import ExplanationVisualizer
from data_loader import load_data_txt, load_heloc, load_kdd_csv, load_ids_csv

def test():
    chosen_attributes = [0, 5]
    clf = RandomForestClassifier(n_jobs=100, n_estimators=50, random_state=5000)
    X, Y = load_data_txt(normalize=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)
    clf.fit(X_train, Y_train)

    explainer = DefaultExplainer(clf, X, [0,5])
    explainer.explain_instance(X_test[18])

    y_clf = clf.predict(X_test)
    y_exp = explainer.sg.surrogate.predict(X_test)
    print('comparison score on test dataset: ', accuracy_score(y_clf, y_exp))

    viz = ExplanationVisualizer(explainer, chosen_attributes)
    viz.present_explanation()

def test_heloc():
    clf = RandomForestClassifier(n_jobs=100, n_estimators=100, random_state=1000)
    X, Y = load_heloc(normalize=False)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1200)
    clf.fit(X_train, Y_train)


    print('accuracy Rf: ', accuracy_score(clf.predict(X_test), Y_test))

    print('total num features', X_train.shape[1])
    explainer = DefaultExplainer(clf, X, list(range(X_train.shape[1])))

    for instance in X_test:
        if clf.predict_proba(instance.reshape(1, -1))[0, 1]  < 0.5:
            explainer.explain_instance(instance)
            break

    viz = ExplanationVisualizer(explainer, None)
    viz.present_explanation()

def runon_kdd():

    mlp_dump_file = "exports/mlp.joblib"
    p = Path(mlp_dump_file)

    X, Y = load_kdd_csv(normalize=True, train=True)


    if p.is_file():
        clf = joblib.load(mlp_dump_file)
    else:
        clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes = (20, 5), random_state = 1)
        clf.fit(X, Y)
        joblib.dump(clf, mlp_dump_file)

    Xtest, Ytest = load_kdd_csv(normalize=True, train=False)
    print('accuracy MLP: ', accuracy_score(clf.predict(Xtest), Ytest))

    explainer = DefaultExplainer(clf, X, None)

    for instance in Xtest:
        if clf.predict(instance.reshape(1, -1)) < 0.5:
            explainer.explain_instance(instance)
            break

    print('counterfact: ', clf.predict(explainer.counterfactual.reshape(1, -1)))
    viz = ExplanationVisualizer(explainer, None)
    viz.present_explanation()

def runon_ids():

    mlp_dump_file = "exports/mlp_ids.joblib"
    p = Path(mlp_dump_file)

    X, Y, names = load_ids_csv()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1200)

    if p.is_file():
        clf = joblib.load(mlp_dump_file)
    else:
        clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes = (20, 5), random_state = 1)
        clf.fit(X_train, Y_train)
        joblib.dump(clf, mlp_dump_file)

    explainer = DefaultExplainer(clf, X, None)

    for instance in X_test:
        if clf.predict(instance.reshape(1, -1)) > 0.5:
            explainer.explain_instance(instance)
            break

    print('accuracy MLP: ', accuracy_score(clf.predict(X_test), Y_test))

    viz = ExplanationVisualizer(explainer, None)
    viz.present_explanation()

if __name__ == '__main__':
    runon_ids()
