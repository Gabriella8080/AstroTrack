import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from astrotrack.preprocess import (
    parse_tle_file,
    filter_tles_by_date,
    load_horizon_profile,
)

# -------------------------
# Helper: minimal valid TLE
# -------------------------
VALID_TLE = (
    "1 25544U 98067A   24100.00000000  .00016717  00000+0  10270-3 0  9000",
    "2 25544  51.6445  23.4362 0007417  43.3475  88.1382 15.50000000 00000",
)

# -------------------------
# parse_tle_file tests
# -------------------------

@pytest.mark.parametrize(
    "file_contents,satcon,expected_len",
    [
        # Standard 2-line TLE
        ("\n".join(VALID_TLE), None, 1),

        # 3LE format with matching constellation
        ("\n".join(["0 STARLINK-TEST", VALID_TLE[0], VALID_TLE[1]]), "STARLINK", 1),

        # 3LE format with non-matching constellation
        ("\n".join(["0 ONEWEB-TEST", VALID_TLE[0], VALID_TLE[1]]), "STARLINK", 0),
    ],
)
def test_parse_tle_file(tmp_path, file_contents, satcon, expected_len):
    file = tmp_path / "tle.txt"
    file.write_text(file_contents)

    result = parse_tle_file(str(file), satcon)

    assert isinstance(result, list)
    assert len(result) == expected_len


# -------------------------
# filter_tles_by_date tests
# -------------------------

@pytest.mark.parametrize(
    "target_date",
    [
        datetime(2024, 4, 10),
        datetime(2023, 1, 1),
    ],
)
def test_filter_tles_by_date_runs(target_date):
    """Test that filtering runs without crashing and returns valid structure."""
    tles = [VALID_TLE]

    result = filter_tles_by_date(tles, target_date)

    assert isinstance(result, list)
    assert all(len(tle) == 2 for tle in result)


# -------------------------
# load_horizon_profile tests (tuple input)
# -------------------------

@pytest.mark.parametrize(
    "azi,elev",
    [
        ([0, 90, 180], [0, 5, 0]),
        ([0, 180], [10, 10]),
        ([45, 135, 225], [1, 2, 3]),
    ],
)
def test_load_horizon_from_tuple(azi, elev):
    azi_out, elev_out = load_horizon_profile((azi, elev))

    assert np.array_equal(azi_out, np.array(azi))
    assert np.array_equal(elev_out, np.array(elev))


# -------------------------
# load_horizon_profile tests (CSV input)
# -------------------------

@pytest.mark.parametrize(
    "columns",
    [
        ("Azimuth", "Elevation"),
        ("azi", "prof"),
        ("Azimuth", "prof"),
    ],
)
def test_load_horizon_from_csv(tmp_path, columns):
    azi_col, elev_col = columns

    file = tmp_path / "horizon.csv"

    df = pd.DataFrame({
        azi_col: [0, 90, 180],
        elev_col: [0, 5, 0],
    })
    df.to_csv(file, index=False)

    azi, elev = load_horizon_profile(str(file))

    assert len(azi) == 3
    assert len(elev) == 3


def test_load_horizon_missing_columns(tmp_path):
    file = tmp_path / "bad.csv"

    df = pd.DataFrame({
        "wrong": [1, 2, 3]
    })
    df.to_csv(file, index=False)

    with pytest.raises(ValueError):
        load_horizon_profile(str(file))