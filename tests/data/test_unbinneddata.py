#  Copyright (c) 2022 zfit
import pickle

import numpy as np
import pytest

import zfit._data.unbinneddata
import zfit._variables.axis as axis
import zfit.z.numpy as znp


@pytest.mark.skip
def test_basic():
    n = 1000
    nobs = 3
    axes = axis.SpaceV2(
        [axis.UnbinnedAxis("x"), axis.UnbinnedAxis("y"), axis.UnbinnedAxis("z")]
    )
    tensor = znp.random.uniform(size=(n, nobs))
    weights = znp.random.uniform(size=(n,))
    data = zfit._data.unbinneddata.UnbinnedData(data=tensor, axes=axes, weights=weights)

    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)
    # assert data.axes == unpickled.axes
    np.testing.assert_array_equal(data.weights, unpickled.weights)
    np.testing.assert_array_equal(data.data, unpickled.data)

    array = data["y"]
    np.testing.assert_array_equal(array, tensor[:, 1])

    # af = asdf.AsdfFile(data.obj_to_repr())
    # with io.BytesIO(b'file') as f:
    #     af.write_to(f)
