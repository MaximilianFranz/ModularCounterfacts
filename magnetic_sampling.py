"""
AUTHOR: Maximilian Franz
Magnetic Sampling is a simple modification of the LAD algorithm to increase sampling speed in the phase, where we are looking for support points.
It is independent of the other implementations as of now, but will be included in the comparison, once it is sufficiently far developed.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
from utils import ct, transform_set, inverse_ct, create_ranges, adjust_features

style.use('ggplot')


class MagneticSampler():
    def __init__(self, clf, scaler, sector_width=0.35, confidence=5, threshold=2):
        """
        Constructor of MagneticSampler
        Args:
            clf: black-box classifier trained on data
            scaler: StandardScaler already fit to data
            sector_width: angle in radians in which to sample per sector
            confidence: number of instances sampled in sector
            threshold: number of instances before aborting due to too many errors
        """
        self.clf = clf
        self.scaler = scaler
        self.sector_width = sector_width
        self.confidence = confidence
        self.threshold = threshold

    @staticmethod
    def sample_grid(
                    num_samples,
                    radius_inner,
                    radius_outer,
                    alphas_lower,
                    alphas_upper,
                    original_instance,
                    restricted=False):
        """

        Samples on a grid generated between linear intervals for each dimension

        This replaces sample_in, as it gets the job done more robustly. It has
        deterministic behaviour and discovers edges more frequently.

        Args:
            num_samples: number of samples to draw in the given sector
            alphas_lower: spherical coordinates between which to sample
                          [0] is radius by convention
            alphas_upper: upper end of coordinate range

        Returns:
        """
        result = np.zeros((1, alphas_lower.size + 1))
        radius_ranges = 1
        samples_per_range = num_samples / radius_ranges
        radius = (radius_inner + radius_outer) / 2
        for i in range(1, radius_ranges + 1):
            # radius = radius_inner + i/radius_ranges*(radius_outer - radius_inner)
            lower = np.append(np.array([radius]), alphas_lower)
            upper = np.append(np.array([radius]), alphas_upper)
            result = np.append(result,
                               create_ranges(lower, upper, samples_per_range).T,
                               axis=0)

        if restricted:
            restr = transform_set(result)
            return adjust_features(original_instance, [0, 5], restr)
        else:
            return transform_set(result) + original_instance

    def get_num_errors(self, samples):
        """ Get number of 'wrong' predictions in a set of predictions
        """

        if self.scaler is None:
            trans_set = samples
        else:
            trans_set = self.scaler.inverse_transform(samples)

        results = self.clf.predict_proba(trans_set)[:, 1]
        return results[results <= 0.5].size

    def clean(self, samples):
        if self.scaler is None:
            trans_set = samples
        else:
            trans_set = self.scaler.inverse_transform(samples)

        prob = self.clf.predict_proba(trans_set)[:, 1]
        pos_res_ind = np.where(prob > 0.5)
        result = samples[pos_res_ind]
        return result

    def magnetic_sampling(self,
                          original_instance,
                          adversarial_instance,
                          num_samples,
                          features,
                          sector_depth=0.6,  # must be set depending on the dataset
                          sector_width=0.35,  # About 20 degree,
                          confidence=10,  # must be set depending on the dataset
                          threshold=3,
                          ):
        """
        magnetic_sampling implemented with restriction to a set of features

        All non-selected features remain fixed

        Args:
            original_instance:
            adversarial_instance:
            num_samples:
            features: list of feature positions in the feature vectors that ought to be used.
            sector_depth:
            sector_width:
            confidence:
            threshold:

        Returns:
            Full instnances created by updating copies of the original_instance at
            the desired features with the new features created through magnetic_sampling
        """
        if self.scaler is not None:
            original_instance = self.scaler.transform(original_instance.reshape(1, -1))[0]
            adversarial_instance = self.scaler.transform(adversarial_instance.reshape(1, -1))[0]

        expand_right = True
        expand_left = True

        restricted_original = original_instance[features]
        restricted_adversarial = adversarial_instance[features]

        found = False
        distance = np.linalg.norm(restricted_adversarial - restricted_original)

        while not found:
            # Prep parameters
            # Note that we work on a transposed space, because we look at the vector between
            # original instance and adversarial_instance, before clf we must redo this step.
            print('distance', distance)
            radius_inner = distance - sector_depth / 2
            radius_outer = distance + sector_depth / 2
            alphas = np.array([inverse_ct(restricted_adversarial - restricted_original)[1:]])
            alphas_lower = alphas - sector_width
            alphas_upper = alphas + sector_width

            # start original sample
            total_samples = np.zeros((1, restricted_original.size))
            while expand_left or expand_right:
                if expand_left:
                    sampled_lower = self.sample_grid(confidence, radius_inner, radius_outer,
                                                     alphas_lower, alphas_lower + sector_width, restricted_original)

                    adjusted = adjust_features(original_instance, features, sampled_lower, restricted_original)
                    errs = self.get_num_errors(adjusted)

                    if errs > threshold:
                        expand_left = False
                    else:
                        alphas_lower -= sector_width
                        total_samples = np.append(total_samples, sampled_lower, axis=0)
                if expand_right:
                    sampled_upper = self.sample_grid(confidence, radius_inner, radius_outer,
                                                     alphas_upper - sector_width, alphas_upper, restricted_original)

                    adjusted = adjust_features(original_instance, features, sampled_upper, restricted_original)
                    errs = self.get_num_errors(adjusted)

                    if errs > threshold:
                        expand_right = False
                    else:
                        alphas_upper += sector_width
                        total_samples = np.append(total_samples, sampled_upper, axis=0)

            total_samples = adjust_features(original_instance, features, total_samples, restricted_original)

            diff = num_samples - total_samples.shape[0]
            print('diff: ', diff)
            if diff > 0:
                # To few samples are drawn
                additional_samples = self.sample_grid(abs(diff), radius_inner, radius_outer,
                                                      alphas_lower, alphas_upper, restricted_original)
                adjusted = adjust_features(original_instance, features, additional_samples, restricted_original)
                total_samples = np.append(total_samples, adjusted, axis=0)

            # Remove edge cases where a negative sample was drawn
            cleaned_samples = self.clean(total_samples)

            diff = num_samples - cleaned_samples.shape[0]

            if diff < 0:
                take = np.random.choice(len(cleaned_samples), num_samples)
                cleaned_samples = cleaned_samples[take]

            print('TOTAL-SHAPE:', total_samples.shape)
            print('CLEAN-SHAPE:', cleaned_samples.shape)

            if cleaned_samples.shape[0] > 0:
                found = True
            else:
                distance *= 2


        if self.scaler is not None:
            return self.scaler.inverse_transform(cleaned_samples)

        return cleaned_samples
