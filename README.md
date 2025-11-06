AstroTrack
===========

**AstroTrack** is a Python package for characterising Low Earth Orbit (LEO) satellite interference in wide-field radio astronomy observations. It provides tools for processing TLEs, selecting flyovers, analysing satellite properties, computing Doppler shifts, generating Power Spectral Density (PSD) plots, and creating animations of satellite trajectories.

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

AstroTrack is structured into several modules, where `preprocess` must be run in order to initialise the satellite data to be used across the other modules complementarily. We provide a brief overview below of the key functions and example usage across all five functional modules.

**1. Preprocessing and Loading Satellite Data (`preprocess`)**

`load_satellite_data()` is the primary function users call for obtaining candidate satellite flyovers subject to the user's analysis strategy. Other functions (e.g. `parse_tle_file`, `filter_tles_by_date`) are helper functions.

```python
from datetime import datetime
from AstroTrack.preprocess import load_satellite_data

data = load_satellite_data(
    tle_file="LEO_TLE_file.txt",
    target_date=datetime(2025, 1, 18, 12, 25, 8),
    obs_len=3600, 
    traj_res=60,
    obs_lat=51.5,
    obs_lon=-0.1,
    R=2000,
    horizon_data="Horizon-Profile.csv", 
    satcon="STARLINK"
)
```

**User Inputs:**

- `tle_file`: Path to a TLE plaintext file, where a single TLE (from [CelesTrak](https://celestrak.org/)) or a LEO TLE catalogue (from [Space-Track](https://www.space-track.org)) are both supported formats.
- `target_date`: Observation date and time as a `datetime` object.
- `obs_len`: Observation duration in seconds.
- `traj_res`: Trajectory resolution in seconds ($\Delta \text{t}$ between positions).
- `obs_lat, obs_lon`: Observer location in degrees (latitude, longitude).
- `R`: Radial constraint in km (distance from observer to include satellites).
- `horizon_data`: Optional CSV file or list defining the local horizon as azimuth-elevation pairs. Use `None` for default baseline flat horizon.
- `satcon`: LEO satellite constellation name as string (e.g. `'OneWeb'`, `'STARLINK'`).

**2. Satellite Filtering & Properties (`satcon_properties`)**

This module is useful for creating subsets of satellite populations and analysing their trajectory properties collectively as well as individually. 

**Flyover Selection:**
- `get_norads(all_sat_data)`: Returns list of satellite NORAD IDs from the loaded dataset.
- `filter_by_norads(all_sat_data, exact_id=None, subset=None)`: Select satellites by NORAD IDs.
- `filter_by_time(all_sat_data, start_time, end_time)`: Select satellites visible in a time range (parameters are `datetime` objects).
- `filter_nth(all_sat_data, n=1)`: Keep every nth satellite from the dataset.
- `filter_custom(all_sat_data, custom_fn)`: Filter using a custom user-defined function.

**Satellite Properties & Plots:**

```python
from AstroTrack.satcon_properties import (
    plot_satellite_trajectory,
    plot_flyover_histogram_by_norad,
    plot_satellite_metric,
    plot_max_elevation_histogram
)
```

```python
plot_satellite_trajectory(
    satellite_data,
    elev: float = 30,
    azim: float = 45,
    time=None,
    ref_points=None,
    show_legend: bool = True,
    figsize=(8, 8),
    font_family="Times New Roman"
)

plot_flyover_histogram_by_norad(
    satellite_data,
    cutoff_norad=None,
    gap_hours: float = 1.0,
    bin_width: int = 500,
    color="lightpink",
    font_family="Times New Roman",
    figsize=(8, 8)
)

plot_satellite_metric(
    satellite_data,
    variable="Elevations",
    threshold=None,
    invert=False,
    font_family="Times New Roman",
    figsize=(8, 8)
)

plot_max_elevation_histogram(
    satellite_data,
    font_family="Times New Roman",
    figsize=(8, 8),
    color="lightpink",
    edgecolor="deeppink"
)
```

**User Inputs**:
- `satellite_data`: Output from `load_satellite_data()`
- Optional plotting parameters: `elev`, `azim`, `time`, `ref_points`, `figsize`, `font_family`, `color`, `bin_width`, `show_legend`.
- Satellite metric selection: `variable` (e.g. `"Elevations"`, `"Distances"`)
- Other thresholds or filtering parameters where applicable.

**3. Doppler Analysis (`doppler_analysis`)**

```python
from AstroTrack.doppler_analysis import (
    plot_doppler_shifts, 
    check_doppler_resolution
)
```
