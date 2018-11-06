"""
AUTHOR: Maximilian Franz
Magnetic Sampling is an extension of the LAD algorithm to increase sampling speed.
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
    Tranforms an array of samples (array of arrays) to cartesian coords
    """
    result = []
    for s in samples:
        result.append(coordinate_transform(s))

    return result


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
    Helper methods to generate linspace for multiple start and endpoinst
    """
    if endpoint==1:
        divisor = num-1
    else:
        divisor = num
    steps = (1.0/divisor) * (stop - start)
    return steps[:,np.newaxis]*np.arange(num) + start[:,np.newaxis]

def magnetic_sampling(predictor_fn,
                    original_instance,
                    adversarial_instance,
                    num_samples,
                    sector_depth=0.4, #must be set depending on the dataset
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
        confidence: Number of samples drawn

    """
    expand_right = True
    expand_left = True

    distance = np.linalg.norm(adversarial_instance - original_instance)
    radius_inner = distance - sector_depth / 2
    radius_outer = distance + sector_depth / 2
    alphas = np.array([inverse_coordinate_transform(adversarial_instance)[1:]])
    alphas_lower = alphas - sector_width
    alphas_upper = alphas + sector_width

    total_samples = np.zeros((1, original_instance.size))

    while expand_left or expand_right:
        if expand_left:
            sampled_lower = sample_in(confidence, radius_inner, radius_outer,
            alphas_lower, alphas_lower + sector_width)
            if get_num_errors(sampled_lower, predictor_fn) > threshold:
                expand_left = False
            else:
                alphas_lower =- sector_width
                total_samples = np.append(total_samples, sampled_lower, axis=0)
        if expand_right:
            sampled_upper = sample_in(confidence, radius_inner, radius_outer,
            alphas_upper - sector_width, alphas_upper)
            if get_num_errors(sampled_upper, predictor_fn) > threshold:
                expand_right = False
            else:
                alphas_upper =+ sector_width
                total_samples= np.append(total_samples, sampled_upper, axis=0)

    diff = num_samples - total_samples.shape[0]
    if diff > 0:
        print('before: ', total_samples)
        additional_samples = sample_in(abs(diff), radius_inner, radius_outer,
         alphas_lower, alphas_upper)
        total_samples = np.append(total_samples, additional_samples, axis=0)
        print('after: ', total_samples)

    if diff > 0:
        # TODO: Implement Choice with masks
        pass

    # TODO: Remove negative sampldes (predictor_fn(sample) < 0.5) from total_samples

    print('TOTAL:', total_samples)
    print('TOTAL-SHAPE:', total_samples.shape)
    return total_samples

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

# Testing
o = np.array([0,0])
a = np.array([1,1])
samples = magnetic_sampling(test_pred, o, a, 20, confidence=30)
plot(samples)
