AstroTrack
===========

**AstroTrack** is a Python package for detecting and characterising **LEO satellite interference** in wide-field radio astronomy observations. It provides tools for processing TLEs, selecting flyovers, analyzing satellite properties, computing Doppler shifts, generating PSD plots, and creating animations of satellite trajectories.

---

Installation
------------

You can install AstroTrack directly from PyPI:

.. code-block:: bash

    pip install AstroTrack

Dependencies (`pip` will be installed automatically if missing):

- numpy
- pandas
- matplotlib
- astropy
- skyfield
- h5py
- tabulate
- tqdm

Optional dependencies for development and testing:

- pytest
- build
- twine

---

Quickstart
----------

AstroTrack is structured into several modules. Below is a brief overview of the key functions and example usage.

**1. Preprocessing and Loading Satellite Data (`preprocess.py`)**

`load_satellite_data()` is the primary function users call for obtaining candidate satellite flyovers subject to a users st. Other functions (e.g., `parse_tle_file`, `filter_tles_by_date`) are helper functions.

```python
from datetime import datetime
from AstroTrack.preprocess import load_satellite_data

data = load_satellite_data(
    tle_file="LEO_TLE_file.txt",
    target_date=datetime(2025, 6, 15, 12, 25, 8),
    obs_len=3600,          # Observation duration (seconds)
    traj_res=60,           # Trajectory resolution (seconds)
    obs_lat=51.5,
    obs_lon=-0.1,
    R=2000,                # Radial constraint (km)
    horizon_data=None,     # CSV file or None
    satcon="OneWeb"
)



