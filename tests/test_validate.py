import pandas as pd
import pytest

from fowsim.data.validate import validate_panel


def test_validate_panel_ok():
    df = pd.DataFrame({
        "iso3": ["PAK", "PAK", "PAK"],
        "year": [2000, 2001, 2002],
        "x": [1.0, 2.0, 3.0],
    })
    validate_panel(df)


def test_validate_panel_missing_cols():
    df = pd.DataFrame({"year": [2000], "x": [1.0]})
    with pytest.raises(ValueError):
        validate_panel(df)
