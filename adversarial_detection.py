from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import gradientgrow
import init
from magnetic_sampling import magnetic_sampling_restricted

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


"""
adversarial detection based on LAD improved by GradientSearch, MagneticSampling
Binary Search on the LAD segments
"""

def get_border_touchpoints(support_points, original_instance, predictor_fn, fineness=5):
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

    return np.array(touchpoints)


def sample_around(border_touchpoints, num_samples, sigma):
    """
    Samples around the border_touchpoints with a normal distribution to generate
    a dataset for training a linear model which yields the explanation

    TODO: Enable restricted mode only sampling in the chosen_attributes
    """
    max_arg = np.amax(border_touchpoints, axis=0)
    min_arg = np.amin(border_touchpoints, axis=0)
    result = np.array(border_touchpoints)
    num_per_point = int(num_samples / len(border_touchpoints))
    for point in border_touchpoints:
        mean = point
        sigmas = (max_arg - min_arg)*0.1
        cov = np.diag(sigmas**2)
        rand = np.random.multivariate_normal(mean, cov, num_per_point)
        result = np.append(result, rand, axis=0)

    return result


def get_first_adversarial(original_instance, predictor, dataset, chosen_attributes):
    """
    Using the GradientSearch approach this method searches the first adversarial
    instance to feed into magnetic_sampling.
    """
    # GradientSearch returns only the two chosen attributes
    dec = gradientgrow.decision(dataset, chosen_attributes, original_instance, predictor)
    dec.gradientSearch(step=0.05, scale=1.0, nsample=100)
    return dec
    # return dec.get_last_instance()

def explain_instance(instance, predictor_fn, dataset, chosen_attributes):
    """
    Using the functions provided in this module this returns a linear
    """
    dec = get_first_adversarial(instance, predictor_fn, dataset, chosen_attributes)
    first_adversarial = dec.get_last_instance()
    path = np.array(dec.GSP)
    print('path: ', path)
    # magnetic_sampling uses the predictor_fn not the predictor, thus pass the corresponding fct
    support_points = magnetic_sampling_restricted(predictor_fn.predict_proba,
                            instance,
                            first_adversarial,
                            num_samples=10,
                            features=chosen_attributes,
                            sector_width=0.00001)

    border_touchpoints = get_border_touchpoints(support_points, instance, predictor_fn)
    X = sample_around(border_touchpoints, 1000, 0.1)

    print('input', X)
    y = predictor_fn.predict_proba(X)[:,1]
    pos = y[y > 0.5]
    neg = y[y <= 0.5]
    print('pred results: ', y)
    print('positive ', pos.size)
    print('negative ', neg.size)

    colors = []
    for y_i in y:
        if y_i > 0.5:
            colors.append('limegreen')
        else:
            colors.append('tomato')
    # plt.scatter(X[:, 0], X[:, 5], c=['limegreen'*int(y[i])+'tomato'*(1-int(y[i])) for i in range(len(y))], cmap = plt.cm.Paired, marker='.', s=25)
    plt.scatter(X[:, 0], X[:, 5], c=colors, cmap = plt.cm.Paired, marker='.', s=25)
    plt.scatter([instance[0]], [instance[5]], s=100, c='blue', marker='X')
    plt.scatter([first_adversarial[0]], [first_adversarial[5]], s=100, c='red', marker='X')
    plt.scatter([support_points[:,0]], [support_points[:,5]], s=100, c='purple', marker='o')
    plt.scatter([border_touchpoints[:,0]], [border_touchpoints[:,5]], s=100, c='black', marker='.')
    plt.scatter(path[:,1], path[:,2], s=100, c='yellow', marker='.')
    plt.xlabel("Attribut " + str(0))
    plt.ylabel("Attribut " + str(5))
    plt.title("Attributraum und dessen Klassifizierung")
    plt.show()

    dec.sectorSearch(fineness=50)


def test():

    X, Y = init.load_data_txt()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)

    # --- Train Random Forest
    predictor = RandomForestClassifier(n_jobs=100, n_estimators=50, random_state=5000)
    predictor.fit(X_train, Y_train)


    # --- Accuracy
    print("Accuracy:", accuracy_score(Y_test, predictor.predict(X_test)))
    print("Report:\n", classification_report(Y_test, predictor.predict(X_test)))

    # chose two parameters to adjust
    chosen_attributes = [0,5]
    print('test instance: ', X_test[14])
    explain_instance(X_test[23], predictor, X, chosen_attributes)

if __name__ == '__main__':
    test()
