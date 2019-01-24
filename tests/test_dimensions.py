import pytest

import zfit
from zfit.core.dimension import is_addable

obs = ['obs' + str(i) for i in range(4)]
space1 = zfit.Space(obs=obs)
space2 = zfit.Space(obs=obs)
space3 = zfit.Space(obs=obs)
space4 = zfit.Space(obs=obs[0:2])


def test_addable():
    with pytest.raises(ValueError):
        assert is_addable(spaces=space1)
    with pytest.raises(ValueError):
        assert is_addable(spaces=[space1])
    with pytest.raises(ValueError):
        assert is_addable(spaces=[])

    assert is_addable(spaces=[space1, space2])
    assert is_addable(spaces=[space1, space2, space3])
    assert not is_addable(spaces=[space1, space2, space3, space4])
