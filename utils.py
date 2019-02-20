import numpy as np
from sklearn.linear_model import RidgeClassifier, lars_path


def ct(spherical):
    """ Tranforms spherical coordinates into a cartesian coordinate vector

    Args:
        spherical: radius, n-2 angles in [0, 2\pi] and the last angle in [0, pi]
    """
    a = np.concatenate((np.array([2 * np.pi]), spherical[1:]))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si * co * spherical[0]


def inverse_ct(coords):
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
        result.append(ct(s))

    return result


def create_ranges(start, stop, num, endpoint=True):
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
        divisor = num - 1
    else:
        divisor = num
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, np.newaxis] * np.arange(num) + start[:, np.newaxis]


def adjust_features(instance, feature_positions, feature_updates, restricted_original):
    """
    Given a complete instance feature vector this method adjusts only the
    features at the specified positions by adding the specified feature_updates
    and creating an array of complete feature vectors with the different changes


    Also transposes the vector back to it position in the original space
    """
    result = np.full((feature_updates.shape[0], instance.size), instance)
    result[:, feature_positions] = feature_updates
    return result

def get_primary_features(data, labels, num_features):
        """ Returns most relevant *num_features* features using lars_path


        Args:
            data: the training data
            labels: labels for training / Y.
            num_features: Number of features desired

        Returns:
            used_features: list of indices of the relevant features in the data
        """
        _, _, coefs = lars_path(data,
                                labels,
                                method='lasso',
                                verbose=False)

        for i in range(len(coefs.T) - 1, 0, -1):
            nonzero = coefs.T[i].nonzero()[0]
            if len(nonzero) <= num_features:
                break
        used_features = nonzero

        return used_features

