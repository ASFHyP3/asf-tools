import numpy as np


def tile_array(a, tile_shape=(200, 200), pad=None):
    """Tile a 2D numpy array

    Turn a 2D numpy array like:
        >>> a = [[0,0,1,1],
                 [0,0,1,1],
                 [2,2,3,3],
                 [2,2,3,3]]
        >>> a.shape
        (4,4)

    into a tiled array like:
        >>> t = tiled_array(a, 2, 2)
        >>> t = [[[0,0],
                  [0,0]],
                 [[1,1],
                  [1,1]],
                 [[2,2],
                  [2,2]],
                 [[3,3],
                  [3,3]]]
        >>> t.shape
        (4, 2, 2)

    Args:
        a: 2D array to tile
        tile_shape: the shape of each tile
        pad: right-bottom pad `a` with `pad` as needed so `a` is evenly divisible into tiles

    Returns:
        tiled: the tiled array
    """
    ar, ac = a.shape
    tr, tc = tile_shape

    if rmod := ar % tr:
        rpad = tr - rmod
    else:
        rpad = 0

    if cmod := ac % tc:
        cpad = tc - cmod
    else:
        cpad = 0

    if rmod or cmod:
        if pad is None:
            raise ValueError(f'Cannot evenly tile a {a.shape} array into ({tr},{tc}) tiles')
        else:
            a = np.pad(a, ((0, rpad), (0, cpad)), constant_values=pad)

    tile_list = []
    for rows in np.vsplit(a, range(tr, ar, tr)):
        tile_list.extend(np.hsplit(rows, range(tc, ac, tc)))
    tiled = np.moveaxis(np.dstack(tile_list), -1, 0)
    return tiled


def untile_array(t, array_shape):
    """Untile a tiled array into a 2D numpy array

    This is the reverse of `tile_array` and will turn a tiled array like:
        >>> t = [[[0,0],
                  [0,0]],
                 [[1,1],
                  [1,1]],
                 [[2,2],
                  [2,2]],
                 [[3,3],
                  [3,3]]]
        >>> t.shape
        (4, 2, 2)

    into a 2D array like:
        >>> a = untile_array(t)
        >>> a = [[0,0,1,1],
                 [0,0,1,1],
                 [2,2,3,3],
                 [2,2,3,3]]
        >>> a.shape
        (4,4)

    Args:
        t: a tiled array
        array_shape: shape to until the array to. If the untiled array's shape is larger
            than this, `untile_array` will assume the original image was right-bottom padded
            to evenly tile, and the padding will be removed.

    Returns:
        untiled: the untiled array
    """
    nt, tr, tc = t.shape
    ar, ac = array_shape

    nr = int(np.ceil(ar / tr))
    nc = int(np.ceil(ac / tc))

    untiled = np.zeros((nr*tr, nc*tc), dtype=t.dtype)

    if ar > untiled.shape[0] or ac > untiled.shape[1]:
        raise ValueError(
            f'array_shape {array_shape} must be the same or smaller than the untiled array {untiled.shape}'
        )

    for ii in range(nr):
        for jj in range(nc):
            untiled[ii*tr:(ii+1)*tr,jj*tc:(jj+1)*tc] = t[ii*nr+jj,:,:]

    return untiled[:ar,:ac]
