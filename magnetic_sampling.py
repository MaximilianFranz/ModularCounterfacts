"""
AUTHOR: Maximilian Franz
Magnetic Sampling is a simple modification of the LAD algorithm to increase sampling speed in the phase, where we are looking for support points.
It is independent of the other implementations as of now, but will be included in the comparison, once it is sufficiently far developed.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class MagneticSampler():

    def __init__(self, clf, scaler, sector_width=0.35, confidence=5, threshold=2):
        self.clf = clf
        self.scaler = scaler
        self.sector_width = sector_width
        self.confidence = confidence
        self.threshold = threshold


    def ct(self, spherical):
        """ Tranforms spherical coordinates into a cartesian coordinate vector

        Args:
            spherical: radius, n-2 angles in [0, 2\pi] and the last angle in [0, pi]
        """
        a = np.concatenate((np.array([2*np.pi]), spherical[1:]))
        si = np.sin(a)
        si[0] = 1
        si = np.cumprod(si)
        co = np.cos(a)
        co = np.roll(co, -1)
        return si*co*spherical[0]

    def inverse_ct(self, coords):
        """
        Tranforms cartesian coordinates into spherical coordinates.

        Naive algorithmic implementation. TODO: Replace with numpy implementation.

        Args:
            coords: Array of cartesian coordinates

        Return:
            alphas: Array of spherical coordinates where the first element is the radius
        """
        radius = np.linalg.norm(coords)

        alphas = [radius]
        for i in range(0, len(coords) - 1):
            arcos = np.arccos(coords[i] / np.linalg.norm(coords[i:]))
            alphas.append(arcos)

        return alphas

    def transform_set(self, samples):
        """
        Tranforms an array of vectors from spherical to cartesian coords
        """
        result = []
        for s in samples:
            result.append(self.ct(s))

        return result

    def sample_grid(self,
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
        result = np.zeros((1,alphas_lower.size + 1))
        radius_ranges = 1
        samples_per_range = num_samples / radius_ranges
        radius = (radius_inner + radius_outer)/2
        for i in range(1, radius_ranges + 1):
            # radius = radius_inner + i/radius_ranges*(radius_outer - radius_inner)
            lower = np.append(np.array([radius]), alphas_lower)
            upper = np.append(np.array([radius]), alphas_upper)
            result = np.append(result,
                            self.create_ranges(lower, upper, samples_per_range).T,
                            axis=0)

        if restricted:
            restr = self.transform_set(result)
            return self.adjust_features(original_instance, [0,5], restr)
        else:
            return self.transform_set(result) + original_instance

    def get_num_errors(self, samples, predictor_fn):
        """ Get number of 'wrong' predictions in a set of predictions
        """

        # TODO: use multi-sample batch mode of predictor_fn!

        if self.scaler is None:
            trans_set = samples
        else:
            trans_set = self.scaler.inverse_transform(samples)

        results = self.clf.predict_proba(trans_set)[:, 1]
        return results[results <= 0.5].size

    def create_ranges(self, start, stop, num, endpoint=True):
        """
        Helper method to generate linspace for multiple start and endpoints

        Args:
            start: start points for the intervals
            stop: stop points for the intervals
            num: number of steps between the single start and stop points
            endpoint: Whether to include the endpoint in the interval
        Returns:

        """
        if endpoint:
            divisor = num-1
        else:
            divisor = num
        steps = (1.0/divisor) * (stop - start)
        return steps[:,np.newaxis]*np.arange(num) + start[:,np.newaxis]


    def adjust_features(self, instance, feature_positions, feature_updates, restricted_original):
        """
        Given a complete instance feature vector this method adjusts only the
        features at the specified positions by adding the specified feature_updates
        and creating an array of complete feature vectors with the different changes


        Also transposes the vector back to it position in the original space
        """
        result = np.full((feature_updates.shape[0], instance.size), instance)
        result[:, feature_positions] = feature_updates
        return result

    def clean(self, samples, predictor_fn):
        if self.scaler is None:
            trans_set = samples
        else:
            trans_set = self.scaler.inverse_transform(samples)

        prob = self.clf.predict_proba(trans_set)[:, 1]
        pos_res_ind = np.where(prob > 0.5)
        result = samples[pos_res_ind]
        return result

        # result = []
        # for sample in samples:
        #     if predictor_fn(sample.reshape(1,-1))[0,1] > 0.5:
        #         result.append(sample)
        #
        # return np.array(result)

    def magnetic_sampling(self, predictor_fn,
                        original_instance,
                        adversarial_instance,
                        num_samples,
                        features,
                        sector_depth=0.6, #must be set depending on the dataset
                        sector_width=0.35, #About 20 degree,
                        confidence=10, #must be set depending on the dataset
                        threshold= 3,
                        ):
        """
        magnetic_sampling implemented with restriction to a set of features

        All non-selected features remain fixed

        Args:
            See magnetic_sampling
            features: list of feature positions in the feature vectors that ought
            to be used.

        Returns:
            Full instnances created by updating copies of the original_instance at
            the desired features with the new features created through magnetic_sampling
        """
        if self.scaler is not None:
            original_instance = self.scaler.transform(original_instance.reshape(1,-1))[0]
            adversarial_instance = self.scaler.transform(adversarial_instance.reshape(1,-1))[0]


        expand_right = True
        expand_left = True

        restricted_original = original_instance[features]
        restricted_adversarial = adversarial_instance[features]

        # Prep parameters
        # Note that we work on a distorted space, because we look at the vector between
        # original instance and adversarial_instance, before clf we must redo this step.
        distance = np.linalg.norm(restricted_adversarial - restricted_original)
        print('distance', distance)
        radius_inner = distance - sector_depth / 2
        radius_outer = distance + sector_depth / 2
        alphas = np.array([self.inverse_ct(restricted_adversarial - restricted_original)[1:]])
        alphas_lower = alphas - sector_width
        alphas_upper = alphas + sector_width

        # start original sample
        total_samples = np.zeros((1, restricted_original.size))
        while expand_left or expand_right:
            if expand_left:
                sampled_lower = self.sample_grid(confidence, radius_inner, radius_outer,
                alphas_lower, alphas_lower + sector_width, restricted_original)

                adjusted = self.adjust_features(original_instance, features, sampled_lower, restricted_original)
                errs = self.get_num_errors(adjusted, predictor_fn)

                if errs > threshold:
                    expand_left = False
                else:
                    alphas_lower = alphas_lower - sector_width
                    total_samples = np.append(total_samples, sampled_lower, axis=0)
            if expand_right:
                sampled_upper = self.sample_grid(confidence, radius_inner, radius_outer,
                alphas_upper - sector_width, alphas_upper, restricted_original)

                adjusted = self.adjust_features(original_instance, features, sampled_upper, restricted_original)
                errs = self.get_num_errors(adjusted, predictor_fn)

                if errs > threshold:
                    expand_right = False
                else:
                    alphas_upper = alphas_upper + sector_width
                    total_samples= np.append(total_samples, sampled_upper, axis=0)


        total_samples = self.adjust_features(original_instance, features, total_samples, restricted_original)

        diff = num_samples - total_samples.shape[0]
        print('diff: ', diff)
        if diff > 0:
            # To few samples are drawn
            additional_samples = self.sample_grid(abs(diff), radius_inner, radius_outer,
             alphas_lower, alphas_upper, restricted_original)
            adjusted = self.adjust_features(original_instance, features, additional_samples, restricted_original)
            total_samples = np.append(total_samples, adjusted, axis=0)

        # Remove edge cases where a negative sample was drawn
        cleaned_samples = self.clean(total_samples, predictor_fn)

        diff = num_samples - cleaned_samples.shape[0]

        if diff < 0:
            take = np.random.choice(len(cleaned_samples), num_samples)
            cleaned_samples = cleaned_samples[take]

        print('TOTAL-SHAPE:', total_samples.shape)
        print('CLEAN-SHAPE:', cleaned_samples.shape)

        if self.scaler is not None:
            return self.scaler.inverse_transform(cleaned_samples)
        return cleaned_samples
