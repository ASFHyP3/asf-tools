from typing import Tuple

import numpy as np

from asf_tools import util


def make_histogram(image):
    image = image.flatten()
    indices = np.nonzero(np.isnan(image))
    image[indices] = 0
    indices = np.nonzero(np.isinf(image))
    image[indices] = 0
    del indices
    size = image.size
    maximum = int(np.ceil(np.amax(image)) + 1)
    # maximum = (np.ceil(np.amax(image)) + 1)
    histogram = np.zeros((1, maximum))
    for i in range(0, size):
        # floor_value = int(np.floor(image[i]))
        floor_value = np.floor(image[i]).astype(np.uint8)
        # floor_value = (np.floor(image[i]))
        if floor_value > 0 and floor_value < maximum - 1:
            temp1 = image[i] - floor_value
            temp2 = 1 - temp1
            histogram[0, floor_value] = histogram[0, floor_value] + temp1
            histogram[0, floor_value - 1] = histogram[0, floor_value - 1] + temp2
    histogram = np.convolve(histogram[0], [1, 2, 3, 2, 1])
    histogram = histogram[2:(histogram.size - 3)]
    histogram = histogram / np.sum(histogram)
    return histogram


def make_distribution(m, v, g, x):
    x = x.flatten()
    m = m.flatten()
    v = v.flatten()
    g = g.flatten()
    y = np.zeros((len(x), m.shape[0]))
    for i in range(0, m.shape[0]):
        d = x - m[i]
        amp = g[i] / np.sqrt(2 * np.pi * v[i])
        y[:, i] = amp * np.exp(-0.5 * (d * d) / v[i])
    return y


def expectation_maximization_threshold(image: np.ndarray, number_of_classes: int, scaling) -> Tuple[np.ndarray]:
    """
    Function for Threshold Calculation using an Expectation Maximization Approach
    """

    image_copy = image.copy()
    image_copy2 = np.ma.filled(image.astype(float), np.nan)  # needed for valid posterior_lookup keys
    image = image.flatten()
    minimum = np.amin(image)
    image = image - minimum + 1
    maximum = np.amax(image)

    size = image.size
    histogram = make_histogram(image)
    nonzero_indices = np.nonzero(histogram)[0]
    histogram = histogram[nonzero_indices]
    histogram = histogram.flatten()
    class_means = (
            (np.arange(number_of_classes) + 1) * maximum /
            (number_of_classes + 1)
    )
    class_variances = np.ones((number_of_classes)) * maximum
    class_proportions = np.ones((number_of_classes)) * 1 / number_of_classes
    sml = np.mean(np.diff(nonzero_indices)) / 1000
    iteration = 0
    while (True):
        class_likelihood = make_distribution(
            class_means, class_variances, class_proportions, nonzero_indices
        )
        sum_likelihood = np.sum(class_likelihood, 1) + np.finfo(
            class_likelihood[0][0]).eps
        log_likelihood = np.sum(histogram * np.log(sum_likelihood))
        for j in range(0, number_of_classes):
            class_posterior_probability = (
                    histogram * class_likelihood[:, j] / sum_likelihood
            )
            class_proportions[j] = np.sum(class_posterior_probability)
            class_means[j] = (
                    np.sum(nonzero_indices * class_posterior_probability)
                    / class_proportions[j]
            )
            vr = (nonzero_indices - class_means[j])
            class_variances[j] = (
                    np.sum(vr * vr * class_posterior_probability)
                    / class_proportions[j] + sml
            )
            del class_posterior_probability, vr
        class_proportions = class_proportions + 1e-3
        class_proportions = class_proportions / np.sum(class_proportions)
        class_likelihood = make_distribution(
            class_means, class_variances, class_proportions, nonzero_indices
        )
        sum_likelihood = np.sum(class_likelihood, 1) + np.finfo(
            class_likelihood[0, 0]).eps
        del class_likelihood
        new_log_likelihood = np.sum(histogram * np.log(sum_likelihood))
        del sum_likelihood
        if ((new_log_likelihood - log_likelihood) < 0.000001):
            break
        iteration = iteration + 1
    del log_likelihood, new_log_likelihood
    class_means = class_means + minimum - 1
    s = image_copy.shape
    posterior = np.zeros((s[0], s[1], number_of_classes))
    posterior_lookup = dict()
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            pixel_val = image_copy2[i, j]
            if pixel_val in posterior_lookup:
                for n in range(0, number_of_classes):
                    posterior[i, j, n] = posterior_lookup[pixel_val][n]
            else:
                posterior_lookup.update({pixel_val: [0] * number_of_classes})
                for n in range(0, number_of_classes):
                    x = make_distribution(
                        class_means[n], class_variances[n], class_proportions[n],
                        image_copy[i, j]
                    )
                    posterior[i, j, n] = x * class_proportions[n]
                    posterior_lookup[pixel_val][n] = posterior[i, j, n]

    ### TODO: MAGIC
    sorti = np.argsort(class_means)
    cms = class_means[sorti]
    cvs = class_variances[sorti]
    cps = class_proportions[sorti]
    xvec = np.arange(cms[0], cms[1], step=.05)
    x1 = make_distribution(cms[0], cvs[0], cps[0], xvec)
    x2 = make_distribution(cms[1], cvs[1], cps[1], xvec)
    dx = np.abs(x1 - x2)
    diff1 = posterior[:, :, 0] - posterior[:, :, 1]
    t_ind = np.argmin(dx)
    return xvec[t_ind] / scaling
    ###  end edits

    # return posterior, class_means, class_variances, class_proportions


def kittler_illingworth_threshold(scene: np.ndarray) -> float:
    """
    The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
    Works on 8-bit scenes only
    Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
    Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
           https://doi.org/10.1016/0031-3203(86)90030-0
    """
    # source: https://github.com/asfadmin/asf-jupyter-notebooks/blob/master/SAR_Training/English/HydroSAR/Lab2-SurfaceWaterExtentMapping.ipynb  # noqa: E501
    hist, bins = np.histogram(scene.ravel(), bins=256)

    c = np.cumsum(hist)
    m = np.cumsum(hist * bins[:-1])
    s = np.cumsum(hist * bins[:-1] ** 2)
    sigma_f = np.sqrt(s / c - (m / c) ** 2)

    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb / cb - (mb / cb) ** 2)

    p = c / c[-1]
    v = p * np.log(sigma_f) + (1 - p) * np.log(sigma_b) - p * np.log(p) - (1 - p) * np.log(1 - p)

    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)

    return bins[idx]
