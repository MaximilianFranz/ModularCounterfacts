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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in rad between vectors 'v1' and 'v2'::

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def coordinate_transform(spherical):
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

def inverse_coordinate_transform(coords):
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

def transform_set(samples):
    """
    Tranforms an array of vectors from spherical to cartesian coords
    """
    result = []
    for s in samples:
        result.append(coordinate_transform(s))

    return result

def sample_grid(num_samples, radius_inner, radius_outer, alphas_lower, alphas_upper, original_instance, restricted=False):
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
        lower = np.append(np.array([radius]), alphas_lower)
        upper = np.append(np.array([radius]), alphas_upper)
        result = np.append(result, create_ranges(lower, upper, samples_per_range).T, axis=0)

    if restricted:
        restr = transform_set(result)
        return adjust_features(original_instance, [0,5], restr)
    else:
        return transform_set(result) + original_instance

def sample_in(num_samples, radius_inner, radius_outer, alphas_lower, alphas_upper):
    """ Samples randomly in the intervals for

    Args:


    Returns:
        numsamples x alphas.shape + 1 sized matrix with spherical coordinates
        in the given intervals
    """
    # Instead of sampling randomly we can arrange a grid with numpy linspace
    # from which we take our samples. This way we cover the whole sector more
    # confidently and it is also deterministic.
    sampled_spherical = []
    for n in range(num_samples):
        vec = np.random.uniform(low=radius_inner, high=radius_outer, size=1)
        vec = np.append(vec, np.random.uniform(low=alphas_lower, high=alphas_upper))
        sampled_spherical.append(vec)
    return transform_set(np.asarray(sampled_spherical))

def get_num_errors(samples, predictor_fn):
    """ Get number of 'wrong' predictions in a set of predictions
    """
    pred = predictor_fn(samples)
    return pred[np.where(pred < 0.5)].size

def create_ranges(start, stop, num, endpoint=True):
    """
    Helper methods to generate linspace for multiple start and endpoints

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

def magnetic_sampling(predictor_fn,
                    original_instance,
                    adversarial_instance,
                    num_samples,
                    sector_depth=0.6, #must be set depending on the dataset
                    sector_width=0.35, #About 20 degree,
                    confidence=10, #must be set depending on the dataset
                    threshold= 3 #must be set depending on confidence (e.g. 30 %)
                    ):
    """
    Args:
        predictor_fn: black-box model that maps to the interval [0,1]
        original_instance: cartesian coordinates of original instance
        adversarial_instance: cartesian coordinates of adversarial instance
        num_samples: Number of samples to be drawn
        sector_depth: radius of the spherical layer (outer minus inner)
        sector_width: angle in radians, applied to all dimensions
        confidence: Number of samples drawn in each sector
        threshold: number of errors tolerated per sector


    """
    expand_right = True
    expand_left = True

    # Prep parameters
    distance = np.linalg.norm(adversarial_instance - original_instance)
    radius_inner = distance - sector_depth / 2
    radius_outer = distance + sector_depth / 2
    alphas = np.array([inverse_coordinate_transform(adversarial_instance - original_instance)[1:]])
    alphas_lower = alphas - sector_width
    alphas_upper = alphas + sector_width

    # start original sample
    total_samples = np.zeros((1, original_instance.size))

    while expand_left or expand_right:
        if expand_left:
            sampled_lower = sample_grid(confidence, radius_inner, radius_outer,
            alphas_lower, alphas_lower + sector_width, original_instance)
            if get_num_errors(sampled_lower, predictor_fn) > threshold:
                expand_left = False
            else:
                alphas_lower = alphas_lower - sector_width
                total_samples = np.append(total_samples, sampled_lower, axis=0)
        if expand_right:
            sampled_upper = sample_grid(confidence, radius_inner, radius_outer,
            alphas_upper - sector_width, alphas_upper, original_instance)
            if get_num_errors(sampled_upper, predictor_fn) > threshold:
                expand_right = False
            else:
                alphas_upper = alphas_upper + sector_width
                total_samples= np.append(total_samples, sampled_upper, axis=0)

    diff = num_samples - total_samples.shape[0]
    if diff > 0:
        # To few samples are drawn
        additional_samples = sample_grid(abs(diff), radius_inner, radius_outer,
         alphas_lower, alphas_upper, original_instance)
        total_samples = np.append(total_samples, additional_samples, axis=0)

    # Remove edge cases where a negative sample was drawn
    cleaned_samples = total_samples[(predictor_fn(total_samples) > 0.5)]

    if diff > 0:
        # To many samples are drawn, thus remove some by random choice
        # TODO: Implement Choice with masks
        pass

    print('TOTAL:', total_samples)
    print('TOTAL-SHAPE:', total_samples.shape)
    print('CLEAN-SHAPE:', cleaned_samples.shape)

    return cleaned_samples

def plot(points):
    """
    Test-Wise helper to plot in 2D case the points and the example-predictor
    """
    x = np.arange(0,2,0.1)
    y = -x + 1.8

    plt.plot(x,y)
    plt.fill_between(x, y, 2, zorder=1)
    plt.scatter(list(points[:,0]),list(points[:,1]), color='black', marker='.', zorder=2)
    plt.show()


def test_pred(samples):
    """
    Args:
        samples: Array of vectors which to evaluate
    """
    def f(x):
        if x[0] + x[1] > 1.8:
            return 0.6
        else:
            return 0.4
    return np.array([f(xi) for xi in samples])


def adjust_features(instance, feature_positions, feature_updates):
    """
    Given a complete instance feature vector this method adjusts only the
    features at the specified positions by adding the specified feature_updates
    and creating an array of complete feature vectors with the different changes
    """
    result = np.array([instance])
    for f in feature_updates:
        new = instance.copy()
        new[feature_positions] += f
        result = np.append(result, [new], axis=0)

    return result


def retrieve_features(instance, feature_positions):
    """
    Given a complete instance feature vector and desired feature_positions this
    method returns a featur vector consi
    """
    return instance[feature_positions]

def magnetic_sampling_restricted(predictor_fn,
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
    expand_right = True
    expand_left = True

    restricted_original = original_instance[features]
    restricted_adversarial = adversarial_instance[features]

    # Prep parameters
    distance = np.linalg.norm(restricted_adversarial - restricted_original)
    print('distance', distance)
    radius_inner = distance - sector_depth / 2
    radius_outer = distance + sector_depth / 2
    alphas = np.array([inverse_coordinate_transform(restricted_adversarial - restricted_original)[1:]])
    alphas_lower = alphas - sector_width
    alphas_upper = alphas + sector_width

    # start original sample
    total_samples = np.zeros((1, restricted_original.size))
    while expand_left or expand_right:
        if expand_left:
            sampled_lower = sample_grid(confidence, radius_inner, radius_outer,
            alphas_lower, alphas_lower + sector_width, restricted_original)

            errs = get_num_errors(adjust_features(original_instance, features, sampled_lower), predictor_fn)

            if errs > threshold:
                expand_left = False
            else:
                alphas_lower = alphas_lower - sector_width
                total_samples = np.append(total_samples, sampled_lower, axis=0)
        if expand_right:
            sampled_upper = sample_grid(confidence, radius_inner, radius_outer,
            alphas_upper - sector_width, alphas_upper, restricted_original)

            errs = get_num_errors(adjust_features(original_instance, features, sampled_upper), predictor_fn)
            if errs > threshold:
                expand_right = False
            else:
                alphas_upper = alphas_upper + sector_width
                total_samples= np.append(total_samples, sampled_upper, axis=0)

    diff = num_samples - total_samples.shape[0]
    if diff > 0:
        # To few samples are drawn
        additional_samples = sample_grid(abs(diff), radius_inner, radius_outer,
         alphas_lower, alphas_upper, restricted_original)
        total_samples = np.append(total_samples, additional_samples, axis=0)

    total_samples = adjust_features(original_instance, features, total_samples)

    # Remove edge cases where a negative sample was drawn
    # cleaned_samples = total_samples[(predictor_fn(total_samples) > 0.5)]
    cleaned_samples = clean(total_samples, predictor_fn)

    if diff > 0:
        # To many samples are drawn, thus remove some by random choice
        # TODO: Implement Choice with masks
        pass

    print('TOTAL-SHAPE:', total_samples.shape)
    print('CLEAN-SHAPE:', cleaned_samples.shape)

    return cleaned_samples

def clean(samples, predictor_fn):
    result = []
    for sample in samples:
        if predictor_fn(sample.reshape(1,-1))[0,1] > 0.5:
            result.append(sample)

    return np.array(result)



# Testing
if __name__ == '__main__':
    # o = np.array([0.5,0.5])
    # a = np.array([1,1])
    # samples = magnetic_sampling_restricted(test_pred, o, a, 20, [0,1], confidence=10)
    # plot(samples)
    mean = [0,0,0,0]
    sigma = np.array([0.1, 0.1, 0.1, 0.1])
    cov = np.diag(sigma**2)
    result = np.random.multivariate_normal(mean, cov, 10)
    print(result)

# print(create_ranges(np.array([1,1,2]), np.array([1,10,5]), 5).T)
