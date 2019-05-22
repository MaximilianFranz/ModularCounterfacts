import numpy as np
from sklearn.linear_model import RidgeClassifier, lars_path

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# noinspection PyPackageRequirements
from matplotlib import style
style.use("ggplot")

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

def sample_normal(border_touchpoints, num_samples, sigma):
    """
    Samples around the border_touchpoints with a normal distribution to generate
    a dataset for training a linear model which yields the explanation

    Normal distribution is parametrized based on the distribution of
    the border_touchpoints, so that we sample along the decision boundary

    """
    max_arg = np.amax(border_touchpoints, axis=0)
    min_arg = np.amin(border_touchpoints, axis=0)

    result = np.array(border_touchpoints)
    num_per_point = int(num_samples / len(border_touchpoints))
    sigmas = (max_arg - min_arg) * sigma

    for point in border_touchpoints:
        mean = point
        cov = np.diag(sigmas ** 2)
        rand = np.random.multivariate_normal(mean, cov, num_per_point)
        result = np.append(result, rand, axis=0)

    return result

def construct_test_data_around_instance(dataset, instance, max_distance=0.3, size=None):
        """
        Sampling instances from the original dataset that are close to the instance given
        :param instance: Around which to sample
        :param max_distance: distance limit within which to sample
        :return:
        """
        data_subset = dataset[np.random.randint(dataset.shape[0], size=20000), :]
        dist = np.sum(np.abs(data_subset - instance), axis=1) / data_subset.shape[1]
        result = data_subset[dist < max_distance]
        if size is None or result.shape[0] < size:
            return result
        else:
            return result[0:size]



def plot_tsne(X, Y, counterfacts):
    data_length = len(X)
    print('positives', len(counterfacts))

    union = np.append(X, counterfacts, axis=0)

    transformer = TSNE()
    x_trans = transformer.fit_transform(union)

    color_map = ['tomato', 'limegreen'] # 0 is attacks / specials, 1 is normal
    plt.scatter(x_trans[0 : data_length, 0], x_trans[0 : data_length, 1], c=Y, cmap=ListedColormap(color_map), s=10, marker=".")
    plt.scatter(x_trans[data_length:, 0], x_trans[data_length:, 1], c='blue', s=10, marker="x")
    plt.show()

    return 0


