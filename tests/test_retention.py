import math

import pytest

from winnow.retention import validate_keep


@pytest.mark.parametrize("value", [0.1, 0.5, 1, "0.25"])
def test_valid_keep(value):
    assert 0 < validate_keep(value) <= 1


@pytest.mark.parametrize("value", [0, -0.1, 1.1, 50, math.inf, math.nan, "50%"])
def test_invalid_keep(value):
    with pytest.raises(ValueError):
        validate_keep(value)
