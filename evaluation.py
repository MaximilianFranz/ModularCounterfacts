import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
import copy

np.random.seed(1)

import init
import gradientgrow
import analysis
import lime
import localsurrogate
from adversarial_detection import AdversarialDetection
from comparison import explainer_evaluation


def finalEvaluation(jobs=103, dataset='uci'):
    # --- Create Training and Test set

    if dataset == 'uci':
        X, Y = init.load_data_txt()
        # For UCI Credit Dataset use LIMIT_BAL (Credit Limit) and AGE
        attr1 = 0
        attr2 = 5
    elif dataset == 'iris':
        X, Y = init.load_data_iris()
        # Use Petal Length and Petal Width to to distinguish best between versicolor and non-versicolor
        attr1 = 2
        attr2 = 3
    elif dataset == 'survival':
        X, Y = init.load_data_survival()
        # TODO : Find best parameter choice with PCA / Lasso
        attr1 = 0
        attr2 = 2
    elif dataset == 'breast_cancer':
        X, Y = init.load_data_breast_cancer()
        # TODO : Find best parameter choice with PCA / Lasso
        attr1 = 20
        attr2 = 23
    else:
        raise NotImplementedError('dataset ' + dataset + 'not available.')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1200)

    # --- Train Random Forest
    clf = RandomForestClassifier(n_jobs=100, n_estimators=50, random_state=5000)
    clf.fit(X_train, Y_train)

    # --- Accuracy
    print("Accuracy:", accuracy_score(Y_test, clf.predict(X_test)))
    print("Report:\n", classification_report(Y_test, clf.predict(X_test)))

    # --- Instances for the evaluation
    # Choose only instances which classify as 'Credit Unworthy'

    test_candidates = np.array(range(0, len(Y_test)))[clf.predict(X_test) == 0][9:jobs]
    # test_candidates = np.delete(test_candidates,
    #                             [9, 16, 17, 18, 22, 24, 25, 29, 32, 33, 36, 37, 66, 67, 71, 76, 78, 82, 87, 88])

    # --- Initialize time-variables
    dt_ls = []
    dt_ls1 = []
    dt_ls2 = []
    dt_gg1 = []
    dt_gg2 = []
    dt_gg3 = []
    dt_gg4 = []
    dt_gg = []
    dt_lime = []
    dt_adv = []

    # --- Initialize accuracy-variables
    acc_ls = []
    acc_gg = []
    acc_lime = []

    min1, max1 = np.min(X[:, attr1]), np.max(X[:, attr1])
    min2, max2 = np.min(X[:, attr2]), np.max(X[:, attr2])

    for i in test_candidates:
        print("################")
        print(" Instance ", i)
        print("################")
        np.random.seed(1)

        positive_found = False
        # TODO: Evaluate difference to previous version in detail!
        # CHECK Git Log
        for j in range(0, 100):
            coord1 = np.random.uniform(min1, max1)
            coord2 = np.random.uniform(min2, max2)
            dummy = copy.copy(list(X_test[i]))
            dummy[attr1] = coord1
            dummy[attr2] = coord2
            coord3 = np.array(clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
            if coord3 >= 0.5:
                positive_found = True
                break

        if positive_found :
            # +--------------+
            # | GradientGrow |
            # +--------------+
            # gradient_grow = gradientgrow.Decision(X, chosen_attr=[attr1, attr2], instance=X_test[i], clf=clf)
            #
            # time0 = time.time()  # --- Starte Zeitmessung
            # time1 = time.time()  # --- Starte Zeitmessung
            # gradient_grow.gradient_search(step=0.05, scale=1.0, nsample=100)
            # time2 = time.time()  # --- Beende Zeitmessung
            # dt_gg1.append(time2 - time1)
            #
            # print('ss-start')
            # time1 = time.time()  # --- Starte Zeitmessung
            # gradient_grow.sector_search(fineness=50)
            # time2 = time.time()  # --- Beende Zeitmessung
            # dt_gg2.append(time2 - time1)
            # print('ss-start')
            #
            # print('localsvm-start')
            # time1 = time.time()  # --- Starte Zeitmessung
            # gradient_grow.svmLocal(nsample=200)
            # time2 = time.time()  # --- Beende Zeitmessung
            # dt_gg3.append(time2 - time1)
            # print('localsvm-end')
            #
            # print('extension-start')
            # time1 = time.time()  # --- Starte Zeitmessung
            # gradient_grow.Extension(limit=20)
            # time2 = time.time()  # --- Beende Zeitmessung
            # dt_gg4.append(time2 - time1)
            # print('extension-end')
            # print('eval-range:', gradient_grow.eval_range)
            #
            # dt_gg.append(time2 - time0)
            #
            # # +----------------+
            # # | LIME-Explainer |
            # # +----------------+
            # print('lime-start')
            # lime_exp = lime.limeExplainer(X, X_test[i], [attr1, attr2], clf)
            #
            # time1 = time.time()  # --- Starte Zeitmessung
            # lime_exp.explain(nsample=200, local=3.0)
            # time2 = time.time()  # --- Beende Zeitmessung
            # dt_lime.append(time2 - time1)
            #
            # print('lime-end')
            #
            # # +-----------------+
            # # | Local Surrogate |
            # # +-----------------+
            # print('localsurrogate-start')
            # local_surrogate_exp = localsurrogate.localSurrogate(X, X_test[i], [attr1, attr2], clf)
            #
            # time0 = time.time()  # --- Starte Zeitmessung
            # time1 = time.time()  # --- Starte Zeitmessung
            # local_surrogate_exp.growingSpheres(nsample=50, eta=1.0)
            # time2 = time.time()  # --- Beende Zeitmessung
            # dt_ls1.append(time2 - time1)
            #
            # time1 = time.time()  # --- Beende Zeitmessung
            # local_surrogate_exp.explain_ls(nsample=200)
            # time2 = time.time()  # --- Beende Zeitmessung
            # dt_ls2.append(time2 - time1)
            #
            # dt_ls.append(time2 - time0)
            # print('localsurrogate-end')


            time1 = time.time()
            # adversarial with Magnetic Sampling
            mce_exp = AdversarialDetection(X, clf=clf, chosen_attributes=[attr1, attr2])
            # mce_exp.explain_instance(X_test[i], num_samples=600)
            time2 = time.time()
            dt_adv.append(time2 - time1)

            mce_exp.explain_auto(X_test[i], num_features=2, num_samples=600)

            # +------------------+
            # | gemeinsamer Plot |
            # +------------------+
            print('ana-start')
            explainer_evaluation(clf,
                                 X_test[i],
                                 [mce_exp],
                     mce_exp.eval_range[:, 0],
                     mce_exp.eval_range[:, 1],
                                 [attr1, attr2])

            break

if __name__ == '__main__':
    finalEvaluation(dataset='survival')

print("\nImplementierung wurde erfolgreich gestartet. Die Methode\n--> finalEvaluation()\nkann nun ausgeführt werden.")
