import numpy as np
import joblib
import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from sacred import Ingredient
from sacred import Experiment
from sacred.observers import MongoObserver

from sklearn.model_selection import train_test_split

# local imports
from data_loader import load_ids_csv, load_kdd_csv
from explainer import DefaultExplainer
from visualizer import ExplanationVisualizer


# Argparsing

EXPERIMENT_NAMES = ['kdd', 'ids', 'uci']




#########################################


ex = Experiment(name='test')
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='sacred'))
#

@ex.config
def my_conf():
    foo = 42

@ex.main
def mymain(_run, foo):
    print('output ', foo)
    for i in range(10):
        _run.log_scalar("some.number", i*10, i)


########################################


ex_ids = Experiment(name='ids')
ex_ids.observers.append(MongoObserver.create(url='localhost:27017', db_name='sacred'))

@ex_ids.config
def ids_conf():
    normalize = True
    random_state = 1203
    chosen_features = None
    mlp_dump_file = "exports/mlp_ids.joblib"
    hidden_layer_sizes = (20, 5)
    classifier = 'mlp'
    quantitative = True


@ex_ids.main
def run_ids(_log,
            normalize,
            random_state,
            chosen_features,
            mlp_dump_file,
            hidden_layer_sizes,
            classifier,
            quantitative
            ):

    p = Path(mlp_dump_file)

    X, Y, names = load_ids_csv(normalize=normalize)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)



    if classifier == "mlp":
        _log.info('using mlp')
        if p.is_file():
            clf = joblib.load(mlp_dump_file)
        else:
            clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
            clf.fit(X_train, Y_train)
            joblib.dump(clf, mlp_dump_file)
    elif classifier == "rf":
        _log.info('using random forest')
        clf = RandomForestClassifier(n_jobs=100, n_estimators=100, random_state=random_state)
        clf.fit(X, Y)

    _log.info('accuracy CLF: ' + str(accuracy_score(clf.predict(X_test), Y_test)))
    print('accuracy CLF: ', accuracy_score(clf.predict(X_test), Y_test))

    pred = clf.predict(X_test)
    false_classified = X_test[pred != Y_test]

    if quantitative:
        explainer = DefaultExplainer(clf, X, chosen_features, features_names=names)
        viz = ExplanationVisualizer(explainer, chosen_features, feature_names=names)
        for instance in false_classified:
            pred = clf.predict(instance.reshape(1, -1))
            explainer.explain_instance(instance, target_label=(1-pred))
            viz.present_explanation(method='relative')
            # viz.present_explanation(method='visual')
            ex_ids.log_scalar('linear_accuracy', viz.linear_score)
            ex_ids.log_scalar('tree_accuracy', viz.tree_score)

    else:
        explainer = DefaultExplainer(clf, X, chosen_features, features_names=names)
        for instance in false_classified:
            if clf.predict(instance.reshape(1, -1)) < 0.5: # explain false positives (attacks (0))
                explainer.explain_instance(instance)
                break


        viz = ExplanationVisualizer(explainer, chosen_features, feature_names=names)


        viz.present_explanation(method='relative')
        viz.present_explanation(method='visual')

        ex_ids.add_artifact('exports/heatmap.png')
        ex_ids.add_artifact('exports/db_tree.pdf')


ex_kdd = Experiment(name='kdd')
ex_kdd.observers.append(MongoObserver.create(url='localhost:27017', db_name='sacred'))

@ex_kdd.config
def kdd_conf():
    normalize = True
    random_state = 1000
    chosen_features = None
    mlp_dump_file = "exports/mlp_kdd.joblib"
    classifier = "mlp"
    quantitative = True


@ex_kdd.main
def run_kdd(_log,
            normalize,
            random_state,
            chosen_features,
            mlp_dump_file,
            classifier,
            quantitative):

    p = Path(mlp_dump_file)

    X, Y, names= load_kdd_csv(normalize=normalize, train=True)

    if classifier == "mlp":
        _log.info('using mlp')
        if p.is_file():
            print('using dumped mlp weights')
            clf = joblib.load(mlp_dump_file)
        else:
            print('training mlp weights')
            clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes = (20, 5), random_state = random_state)
            clf.fit(X, Y)
            joblib.dump(clf, mlp_dump_file)
    elif classifier == "rf":
        _log.info('using random forest')
        clf = RandomForestClassifier(n_jobs=100, n_estimators=100, random_state=random_state)
        clf.fit(X, Y)


    Xtest, Ytest, names = load_kdd_csv(normalize=normalize, train=False)
    _log.info('accuracy CLF: ' + str(accuracy_score(clf.predict(Xtest), Ytest)))
    print('accuracy CLF: ', accuracy_score(clf.predict(Xtest), Ytest))


    pred = clf.predict(Xtest)
    false_classified = Xtest[pred != Ytest]

    if quantitative:
        explainer = DefaultExplainer(clf, X, chosen_features, features_names=names)
        viz = ExplanationVisualizer(explainer, chosen_features, feature_names=names)
        for instance in Xtest[:20, : ]:
            pred = clf.predict(instance.reshape(1, -1))
            if pred < 0.5:
                explainer.explain_instance(instance, target_label=(1-pred))
                viz.present_explanation(method='relative')
                ex_kdd.log_scalar('linear_accuracy', viz.linear_score)
                ex_kdd.log_scalar('tree_accuracy', viz.tree_score)
    else:
        explainer = DefaultExplainer(clf, X, chosen_features, features_names=names)
        for instance in Xtest:
            if clf.predict(instance.reshape(1, -1)) < 0.5: # try on false positive (wrongly classfied as attack)
                explainer.explain_instance(instance)
                break

        viz = ExplanationVisualizer(explainer, chosen_features, feature_names=names)
        viz.present_explanation(method='relative')
        # viz.present_explanation(method='visual')

        ex_kdd.add_artifact('exports/heatmap.png')
        ex_kdd.add_artifact('exports/db_tree.pdf')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--ex", dest='experiment', help="choose which experiment to run", choices=EXPERIMENT_NAMES, default='ids')
    args = parser.parse_args()


    if args.experiment == 'kdd':
        ex_kdd.run()
    if args.experiment == 'ids':
        ex_ids.run()
    if args.experiment == 'uci':
        pass
        # ex_uci.run()
