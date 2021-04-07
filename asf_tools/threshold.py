import numpy as np


def kittler_illingworth_threshold(scene: np.ndarray) -> float:
    """
    The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
    Works on 8-bit scenes only
    Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
    Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
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
