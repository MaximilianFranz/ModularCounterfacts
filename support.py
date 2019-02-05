"""

"""
from magnetic_sampling import MagneticSampler
from sklearn.preprocessing import StandardScaler


class SupportFinder():
    def __init__(self, clf, data, chosen_attributes):
        self.clf = clf
        self.data = data
        self.scaler = StandardScaler().fit(data)
        self.chosen_attributes = chosen_attributes

    def support_points_with_magnetic_sampling(self,
                                              instance,
                                              counterfactual,
                                              num_support=10,
                                              sector_width=0.35,
                                              confidence=5,
                                              threshold=2):
        """
        Nelder-Mead and Magnetic Sampling

        Args:
            instance:
            num_support:
            sector_width:
            confidence:
            threshold:

        Returns:

        """

        ms = MagneticSampler(self.clf, scaler=self.scaler)  # using default params for now

        support_points = ms.magnetic_sampling(instance,
                                              counterfactual,
                                              num_support=num_support,
                                              features=self.chosen_attributes,
                                              sector_width=sector_width,
                                              confidence=confidence,
                                              threshold=threshold
                                              )
        return support_points