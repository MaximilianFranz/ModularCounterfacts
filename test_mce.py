from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from explainer import DefaultExplainer
from visualizer import ExplanationVisualizer
from init import load_data_txt

def test():
    chosen_attributes = [0, 5]
    clf = RandomForestClassifier(n_jobs=100, n_estimators=50, random_state=5000)
    X, Y = load_data_txt(normalize=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)
    clf.fit(X_train, Y_train)

    explainer = DefaultExplainer(clf, X, chosen_attributes)
    explainer.explain_instance(X_test[18])
    viz = ExplanationVisualizer(explainer, chosen_attributes)
    viz.present_explanation()

if __name__ == '__main__':
    test()
