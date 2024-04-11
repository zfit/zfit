#  Copyright (c) 2024 zfit

import copy

import numpy as np
import pandas
import pandas as pd
import pytest
import tensorflow as tf
import uproot

import zfit

obs1 = ("obs1", "obs2", "obs3")


@pytest.fixture
def example_data1():
    return np.random.random(size=(7, len(obs1)))


@pytest.fixture
def obs3d():
    return obs1


@pytest.fixture
def np_data1(obs3d):
    return np.random.random(size=(7, len(obs3d)))


class TestHashPDF(zfit.pdf.BasePDF):
    def __init__(self, obs, lasthash):
        super().__init__(obs=obs)
        self.lasthash = lasthash

    def _unnormalized_pdf(self, x):
        import zfit.z.numpy as znp

        self.lasthash = x.hashint
        return znp.abs(x.unstack_x()[0])


@pytest.fixture
def data1(obs3d, np_data1):
    return zfit.Data.from_numpy(obs=obs3d, array=np_data1)


@pytest.fixture
def space2d():
    obs = ["obs1", "obs2"]
    lower1, lower2 = -1, -2
    upper1, upper2 = 1.9, 3.6
    space1 = zfit.Space(obs[0], limits=(lower1, upper1))
    space2 = zfit.Space(obs[1], limits=(lower2, upper2))
    space2d = space1 * space2
    return space2d


@pytest.mark.parametrize("obs_alias", [None, {"pt1": "pt2", "pt2": "pt1"}])
def test_from_root_limits(obs_alias):
    from skhep_testdata import data_path

    path_root = data_path("uproot-Zmumu.root")

    branches = ["pt1", "pt2"]
    weight_branch = "eta1"
    with uproot.open(path_root) as f:
        tree = f["events"]
        true_data_uncut = tree.arrays(branches + [weight_branch], library="pd")
    lower1 = 40.5
    upper1 = 60.2
    lower2 = 10.5
    upper2 = 40.4
    if obs_alias is not None:  # swap branches in true_data_uncut
        true_data_uncut = true_data_uncut.rename(
            {"pt1": "pt2", "pt2": "pt1"}, axis="columns"
        )

    true_data = true_data_uncut.query(
        f"pt1 > {lower1} & pt1 < {upper1} & pt2 > {lower2} & pt2 < {upper2}"
    )
    true_weights = true_data.pop(weight_branch)
    obs1 = zfit.Space("pt1", limits=(lower1, upper1))
    obs2 = zfit.Space("pt2", limits=(lower2, upper2))
    obs = obs1 * obs2

    data = zfit.Data.from_root(
        path=path_root,
        treepath="events",
        obs=obs,
        weights=weight_branch,
        obs_alias=obs_alias,
    )
    x_np = data.value()
    np.testing.assert_allclose(x_np, true_data[branches].values)
    np.testing.assert_allclose(data.weights, true_weights)


@pytest.mark.parametrize(
    "weights_factory",
    [
        lambda n=None: None,
        lambda n=1000: 2.0 * tf.ones(shape=(n,), dtype=tf.float64),
        lambda n=1000: np.random.normal(size=n),
        lambda n=None: "eta1",
    ],
)
def test_from_root(weights_factory):
    from skhep_testdata import data_path

    path_root = data_path("uproot-Zmumu.root")

    branches = ["pt1", "pt2", "phi2"]
    with uproot.open(path_root) as f:
        tree = f["events"]
        true_data = tree.arrays(library="pd")
    weights = weights_factory(n=true_data.shape[0])
    data = zfit.Data.from_root(
        path=path_root, treepath="events", obs=branches, weights=weights
    )
    x = data.value()
    if weights is not None:
        weights_np = data.weights
    else:
        weights_np = weights
    np.testing.assert_allclose(x, true_data[branches].values)
    if weights is not None:
        true_weights = (
            weights if not isinstance(weights, str) else true_data[weights].values
        )
        np.testing.assert_allclose(weights_np, true_weights)
    else:
        assert weights_np is None


@pytest.mark.parametrize(
    "weights_factory",
    [
        lambda: None,
        lambda: 2.0 * tf.ones(shape=(1000,), dtype=tf.float64),
        lambda: np.random.normal(size=1000),
    ],
)
@pytest.mark.parametrize("init", [True, False], ids=["init", "from_numpy"])
def test_from_numpy(weights_factory, obs3d, init):
    weights = weights_factory()

    example_data = np.random.random(size=(1000, len(obs1)))
    if not init:
        data = zfit.Data.from_numpy(obs=obs3d, array=example_data, weights=weights)
    else:
        data = zfit.Data(example_data, obs=obs3d, weights=weights)
    x = data.value()
    weights_from_data = data.weights
    x_np = x
    np.testing.assert_array_equal(example_data, x_np)
    if weights is not None:
        np.testing.assert_allclose(weights_from_data, weights)
    else:
        assert weights_from_data is None


@pytest.mark.parametrize('init', [True, False], ids=['init', 'from_pandas'])
def test_from_to_pandas(obs3d, init):
    dtype = np.float32
    example_data_np = np.random.random(size=(1000, len(obs3d)))
    example_weights = np.random.random(size=(1000,))
    example_data = pd.DataFrame(data=example_data_np, columns=obs3d)
    if init:
        data = zfit.Data(example_data, obs=obs3d, dtype=dtype)
    else:
        data = zfit.Data.from_pandas(obs=obs3d, df=example_data, dtype=dtype)
    x = data.value()
    assert x.dtype == dtype
    assert x.dtype == dtype
    np.testing.assert_array_equal(example_data_np.astype(dtype=dtype), x)

    # test auto obs retreavel
    example_data2 = pd.DataFrame(data=example_data_np, columns=obs3d)
    if init:
        data2 = zfit.Data(example_data2)
    else:
        data2 = zfit.Data.from_pandas(df=example_data2)
    assert data2.obs == obs3d
    x2 = data2.value()
    np.testing.assert_array_equal(example_data_np, x2)
    assert len(data2.obs) == len(obs3d)
    assert len(data2.obs) == len(data2.to_pandas().columns)

    data2w = zfit.Data.from_pandas(df=example_data2, weights=example_weights)
    df2w = data2w.to_pandas()
    if init:
        data2w2 = zfit.Data(df2w)
    else:
        data2w2 = zfit.Data.from_pandas(df=df2w)
    assert data2w2.obs == obs3d
    np.testing.assert_allclose(data2w2.weights, example_weights)
    np.testing.assert_allclose(data2w2.value(), example_data_np)
    df = data2.to_pandas()
    pandas.testing.assert_frame_equal(df, example_data)


@pytest.mark.parametrize("weights_as_branch", [True, False], ids=["weights_as_branch", "weights_as_array"])
@pytest.mark.parametrize("init", [True, False], ids=["init", "from_tensor"])
def test_from_pandas_limits(weights_as_branch, obs3d, init):
    from skhep_testdata import data_path

    path_root = data_path("uproot-Zmumu.root")

    branches = ["pt1", "pt2"]
    weight_branch = "eta1"
    with uproot.open(path_root) as f:
        tree = f["events"]
        true_data_uncut = tree.arrays(branches + [weight_branch], library="pd")
    lower1 = 23.5
    upper1 = 49.2
    lower2 = 21.5
    upper2 = 44.4
    true_data = true_data_uncut.query(
        f"pt1 > {lower1} & pt1 < {upper1} & pt2 > {lower2} & pt2 < {upper2}"
    )

    true_weights = true_data.pop(weight_branch)
    obs3d = zfit.Space("pt1", limits=(lower1, upper1))
    obs2 = zfit.Space("pt2", limits=(lower2, upper2))
    obs = obs3d * obs2

    weights_for_pandas = (
        true_data_uncut[weight_branch] if weights_as_branch else weight_branch
    )

    if init:
        data = zfit.Data(true_data_uncut, obs=obs, weights=weights_for_pandas)
    else:
        data = zfit.Data.from_pandas(
            obs=obs, df=true_data_uncut, weights=weights_for_pandas
        )
    x = data.value()
    np.testing.assert_allclose(x, true_data[branches].values)
    np.testing.assert_allclose(data.weights, true_weights)


@pytest.mark.parametrize(
    "weights_factory",
    [
        lambda: None,
        lambda: 2.0 * tf.ones(shape=(1000,), dtype=tf.float64),
        lambda: np.random.normal(size=1000),
    ],
)
@pytest.mark.parametrize("init", [True, False], ids=["init", "from_tensor"])
def test_from_tensors(weights_factory, init):
    weights = weights_factory()
    true_tensor = 42.0 * tf.ones(shape=(1000, 1), dtype=tf.float64)

    if init:
        data = zfit.Data.from_tensor(obs="obs1", tensor=true_tensor, weights=weights)
    else:
        data = zfit.Data(true_tensor, obs="obs1", weights=weights)

    weights_data = data.weights
    x = data.value()

    np.testing.assert_allclose(x, true_tensor)
    if weights is not None:
        weights_data = weights_data
        np.testing.assert_allclose(weights_data, weights)
    else:
        assert weights is None

    df = data.to_pandas()
    data_new = zfit.Data.from_pandas(df=df)
    assert data_new.obs == data.obs
    np.testing.assert_allclose(data_new.value(), data.value())
    if weights is not None:
        np.testing.assert_allclose(data_new.weights, data.weights)
    else:
        assert data_new.weights is None


def test_overloaded_operators(data1):
    with pytest.raises(TypeError):
        a = data1 * 5.0


def test_sort_by_obs(data1, obs3d):
    new_obs = (obs3d[1], obs3d[2], obs3d[0])
    example_data1 = np.array(data1.value())
    new_array = copy.deepcopy(example_data1)[:, np.array((1, 2, 0))]
    # new_array = np.array([new_array[:, 1], new_array[:, 2], new_array[:, 0]])
    assert data1.obs == obs3d, "If this is not True, then the test will be flawed."
    data1new = data1.with_obs(new_obs)
    assert data1new.obs == new_obs
    np.testing.assert_array_equal(new_array, data1new.value())
    new_array2 = copy.deepcopy(new_array)[:, np.array((1, 2, 0))]
    # new_array2 = np.array([new_array2[:, 1], new_array2[:, 2], new_array2[:, 0]])
    new_obs2 = (new_obs[1], new_obs[2], new_obs[0])
    data1new2 = data1new.with_obs(new_obs2)
    assert data1new2.obs == new_obs2
    np.testing.assert_array_equal(new_array2, data1new2.value())

    assert data1new.obs == new_obs

    assert data1.obs == obs3d
    np.testing.assert_array_equal(example_data1, data1.value())


def test_data_axis_access(obs3d, data1):
    import zfit.z.numpy as znp

    true_mapping = {
        obs: znp.reshape(arr, (-1, 1))
        for obs, arr in zip(obs3d, znp.transpose(data1.value()))
    }
    for obs in obs3d:
        np.testing.assert_allclose(data1.value(obs), true_mapping[obs][:, 0])
        np.testing.assert_allclose(data1.value([obs]), true_mapping[obs])
        np.testing.assert_allclose(data1[obs], true_mapping[obs][:, 0])
        np.testing.assert_allclose(data1[[obs]], true_mapping[obs])
    obs2d = [obs3d[2], obs3d[1]]
    array2d = znp.concatenate([true_mapping[obs] for obs in obs2d], axis=1)
    np.testing.assert_allclose(data1[obs2d], array2d)
    np.testing.assert_allclose(data1.value(obs2d), array2d)


def test_subdata(obs3d, data1):
    new_obs = (obs3d[0], obs3d[1])
    new_array = np.array(data1.value())[:, np.array((0, 1))]
    # new_array = np.array([new_array[:, 0], new_array])
    data1new = data1.with_obs(obs=new_obs)
    assert data1new.obs == new_obs
    np.testing.assert_array_equal(new_array, data1new.value())
    new_array2 = np.array(new_array)[:, 1]
    # new_array2 = np.array([new_array2[:, 1]])
    new_obs2 = (new_obs[1],)
    data1new2 = data1new.with_obs(new_obs2)
    assert data1new2.obs == new_obs2
    np.testing.assert_array_equal(new_array2, data1new2.value()[:, 0])

    # with pytest.raises(ValueError):
    #     with data1.sort_by_obs(obs=new_obs):
    #         data1.value()

    assert data1.obs == obs3d
    np.testing.assert_array_equal(data1.value(), data1.value())


@pytest.mark.parametrize(
    "weights_factory",
    [
        lambda: None,
        lambda: np.random.normal(size=5),
    ],
    ids=["no_weights", "weights"],
)
def test_data_range(weights_factory):
    data1 = np.array([[1.0, 2], [0, 1], [-2, 1], [-1, -1], [-5, 10]])
    # data1 = data1.transpose()
    obs = ["obs1", "obs2"]
    lower1, lower2 = (0.5, 1), (-3, -2)
    upper1, upper2 = (1.5, 2.5), (-1.5, 1.5)
    space1 = zfit.Space(obs, limits=(lower1, upper1))
    space2 = zfit.Space(obs, limits=(lower2, upper2))
    data_range = space1 + space2
    cut_data1 = data1[np.array((0, 2)), :]

    weights = weights_factory()
    if weights is not None:
        cut_weights = weights[np.array((0, 2))]

    dataset = zfit.Data.from_tensor(obs=obs, tensor=data1, weights=weights)
    value_uncut = dataset.value()
    np.testing.assert_equal(data1, value_uncut)
    dataset_cut = dataset.with_obs(data_range)
    # with dataset.set_data_range(data_range):
    value_cut = dataset_cut.value()
    np.testing.assert_equal(cut_data1, value_cut)
    if dataset_cut.has_weights:
        np.testing.assert_equal(cut_weights, dataset_cut.weights)
    np.testing.assert_equal(
        data1, value_uncut
    )  # check  that the original did NOT change

    np.testing.assert_equal(cut_data1, value_cut)
    np.testing.assert_equal(data1, dataset.value())


def test_multidim_data_range():
    nevents1 = 11
    neventstrue1 = 6
    data1 = np.linspace((0, 5), (10, 15), num=nevents1)
    data_true = np.linspace(10, 15, num=neventstrue1)
    lower = ((5, 5),)
    upper = ((10, 15),)
    obs1 = "x"
    obs2 = "y"
    data_range = zfit.Space([obs1, obs2], limits=(lower, upper))
    dataset = zfit.Data.from_numpy(array=data1, obs=data_range)
    xdata = dataset.unstack_x("x")
    assert tf.is_tensor(xdata)
    xdatalist = dataset.unstack_x("x", always_list=True)
    assert len(xdatalist) == 1
    assert isinstance(xdatalist, list)
    assert tf.is_tensor(xdatalist[0])
    yxdata = dataset.unstack_x(["y", "x"])
    assert len(yxdata) == 2
    assert tf.is_tensor(yxdata[0])
    assert xdata.shape == (neventstrue1,)
    assert dataset.nevents == 6
    datasetnew1 = dataset.with_obs(obs=obs1)
    assert datasetnew1.nevents == 6
    assert dataset.nevents == 6
    datasetnew2 = dataset.with_obs(obs=obs2)
    assert datasetnew2.nevents == 6
    np.testing.assert_allclose(data_true, datasetnew2.value()[:, 0])

    data2 = np.linspace((0, 5), (10, 15), num=11)[:, 1]
    data_range = zfit.Space([obs1], limits=(5, 15))
    dataset = zfit.Data.from_numpy(array=data2, obs=data_range)
    assert dataset.nevents == 11


def test_data_hashing(space2d):
    npdata1 = np.random.uniform(size=(3352, 2))
    # data1 = data1.transpose()
    data1 = zfit.Data.from_numpy(obs=space2d, array=npdata1, use_hash=True)
    assert data1.hashint is not None
    testhashpdf = TestHashPDF(obs=space2d, lasthash=data1.hashint)
    assert testhashpdf.lasthash == data1.hashint
    oldhashint = data1.hashint
    data1 = data1.with_weights(np.random.uniform(size=data1.nevents))
    assert data1.hashint is not None
    assert oldhashint != data1.hashint
    assert data1.hashint != testhashpdf.lasthash
    assert oldhashint == testhashpdf.lasthash
    testhashpdf.pdf(data1, norm=False)
    assert oldhashint != testhashpdf.lasthash
    assert data1.hashint == testhashpdf.lasthash

    with zfit.run.set_graph_mode(
            True
    ):  # meaning integration is now done in graph and has "None"
        oldhashint = data1.hashint
        data2 = data1.with_weights(np.random.uniform(size=data1.nevents))
        testhashpdf.pdf(data2)
        assert oldhashint != testhashpdf.lasthash


def test_hashing_resample(space2d):
    n = 1534
    pdf = zfit.pdf.Gauss(obs=space2d.with_obs(space2d.obs[0]), mu=0.4, sigma=0.8)
    sample = pdf.create_sampler(n)
    sample._use_hash = True
    hashint = sample.hashint
    sample.resample()
    assert sample.hashint != hashint
