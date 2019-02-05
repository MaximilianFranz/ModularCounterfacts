"""
Maybe split into find first counterfactual and gather support points, as one could possibly omit the support points
and only work with a single
"""
import numpy as np


class BoundaryFinder():
    def __init__(self, clf):
        self.clf = clf

    def touchpoints_using_binary_search(self, support_points, original_instance, fineness=5):
        """
        Uses a sort of binary search to find the border touchpoint on segments faster.
        """
        touchpoints = []
        for point in support_points:
            l = 0
            r = 1
            for i in range(fineness):
                m = (r + l) / 2.0
                x_m = original_instance + m * (point - original_instance)
                if self.clf.predict_proba(np.array(x_m).reshape(1, -1))[0, 1] <= 0.5:
                    l = m
                else:
                    r = m
            touchpoints.append(x_m)

        return np.array(touchpoints)

    def sector_search(self, counterfactual, original, triangular_point, clf, ):
        """

        Adapt sector_search from gradientgrow to work here.

        Args:
            counterfactual:
            original:
            triangular_point:
            clf:

        Returns:

        """
        pass
