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

AstroTrack is structured into several modules, where `preprocess.py` must be run in order to initialise the satellite data to be used across the other modules complementarily. We provide a brief overview below of the key functions and example usage across all five functional modules.

**1. Preprocessing and Loading Satellite Data (`preprocess.py`)**

`load_satellite_data()` is the primary function users call for obtaining candidate satellite flyovers subject to the user's analysis strategy. Other functions (e.g. `parse_tle_file`, `filter_tles_by_date`) are helper functions.

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

User Inputs
-----------

- **tle_file**: Path to a TLE file. Can be a single TLE text file (2 lines per satellite) or a catalogue from CelesTrak.
- **target_date**: Observation date and time as a `datetime` object (e.g., `datetime(2025,6,15,12,25,8)`).
- **obs_len**: Observation duration in seconds (default 3600 s = 1 hour).
- **traj_res**: Trajectory resolution in seconds (time step between predicted positions).
- **obs_lat, obs_lon**: Observer location in degrees (latitude, longitude).
- **R**: Radial constraint in km (distance from observer to include satellites).
- **horizon_data**: Optional CSV file or list defining the local horizon (azimuth vs elevation). Use `None` for automatic calculation.
- **satcon**: Satellite constellation name as a string (e.g., `'OneWeb'`, `'STARLINK'`).


