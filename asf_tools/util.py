from typing import Tuple

import numpy as np


def tile_array(array: np.ndarray, tile_shape: Tuple[int, int] = (200, 200), pad_value: float = None):
    """Tile a 2D numpy array

    Turn a 2D numpy array like:
        >>> array = [[0, 0, 1, 1],
        ...          [0, 0, 1, 1],
        ...          [2, 2, 3, 3],
        ...          [2, 2, 3, 3]]
        >>> array.shape
        (4, 4)

    into a tiled array like:
        >>> tiles = tiled_array(array, 2, 2)
        >>> print(tiles)
        [[[0, 0],
          [0, 0]],
         [[1, 1],
          [1, 1]],
         [[2, 2],
          [2, 2]],
         [[3, 3],
          [3, 3]]]
        >>> tiles.shape
        (4, 2, 2)

    Args:
        array: 2D array to tile
        tile_shape: the shape of each tile
        pad_value: right-bottom pad `a` with `pad` as needed so `a` is evenly divisible into tiles

    Returns:
        the tiled array
    """
    array_rows, array_columns = array.shape
    tile_rows, tile_columns = tile_shape

    if rmod := array_rows % tile_rows:
        rpad = tile_rows - rmod
    else:
        rpad = 0

    if cmod := array_columns % tile_columns:
        cpad = tile_columns - cmod
    else:
        cpad = 0

    if rpad or cpad:
        if pad_value is None:
            raise ValueError(f'Cannot evenly tile a {array.shape} array into ({tile_rows},{tile_columns}) tiles')
        else:
            array = np.pad(array, ((0, rpad), (0, cpad)), constant_values=pad_value)

    tile_list = []
    for rows in np.vsplit(array, range(tile_rows, array_rows, tile_rows)):
        tile_list.extend(np.hsplit(rows, range(tile_columns, array_columns, tile_columns)))
    tiled = np.moveaxis(np.dstack(tile_list), -1, 0)
    return tiled


def untile_array(tiled_array, array_shape: Tuple[int, int]):
    """Untile a tiled array into a 2D numpy array

    This is the reverse of `tile_array` and will turn a tiled array like:
        >>> tiled_array = [[[0,0],
        ...                 [0,0]],
        ...                [[1,1],
        ...                 [1,1]],
        ...                [[2,2],
        ...                 [2,2]],
        ...                [[3,3],
        ...                 [3,3]]]
        >>> tiled_array.shape
        (4, 2, 2)

    into a 2D array like:
        >>> array = untile_array(tiled_array)
        >>> print(array)
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [2, 2, 3, 3],
         [2, 2, 3, 3]]
        >>> array.shape
        (4, 4)

    Args:
        tiled_array: a tiled array
        array_shape: shape to untile the array to. If the untiled array's shape is larger
            than this, `untile_array` will assume the original image was right-bottom padded
            to evenly tile, and the padding will be removed.

    Returns:
        the untiled array
    """
    _, tile_rows, tile_columns = tiled_array.shape
    array_rows, array_columns = array_shape

    untiled_rows = int(np.ceil(array_rows / tile_rows))
    untiled_columns = int(np.ceil(array_columns / tile_columns))

    untiled = np.zeros((untiled_rows*tile_rows, untiled_columns*tile_columns), dtype=tiled_array.dtype)

    if (array_size := array_rows * array_columns) > tiled_array.size:
        raise ValueError(
            f'array_shape {array_shape} will result in an array bigger than the tiled array:'
            f' {array_size} > {tiled_array.size}'
        )

    for ii in range(untiled_rows):
        for jj in range(untiled_columns):
            untiled[ii*tile_rows:(ii+1)*tile_rows, jj*tile_columns:(jj+1)*tile_columns] = \
                tiled_array[ii * untiled_rows + jj, :, :]

    return untiled[:array_rows, :array_columns]
