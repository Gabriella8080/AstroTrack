AstroTrack
===========

**AstroTrack** is a Python package for characterising Low Earth Orbit (LEO) satellite interference in wide-field radio astronomy observations. It provides tools for processing TLEs, selecting flyovers, analysing satellite properties, computing Doppler shifts, generating Power Spectral Density (PSD) plots, and creating animations of satellite trajectories.

---

Installation
------------

You can install **AstroTrack** directly from PyPI:

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

**AstroTrack** is organised into five core modules, where the workflow begins `preprocess` to initialise and structure the satellite data for analysis. The other subsequent modules build up on this to derive orbital properties, Doppler behaviour with respect to a ground-based experiment, animations, and enabling spectral associations.

We provide a brief overview below of each module, their key functions and example usage. 


**1. Preprocessing and Loading Satellite Data (`preprocess`)**

This module handles the initial preparation of the satellite orbital data. It parses TLE catalogues from a plaintext document, applies filtering, and generates a structured dataset for later analysis. 

`load_satellite_data()` is the primary function, returning preprocessed satellite data customised to a user's preferences and ready for use across all other modules.

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
- `horizon_data`: Optional CSV file or list defining the local horizon as azimuth-elevation pairs. Use `baseline_horizon.csv` for default baseline flat horizon.
- `satcon`: LEO satellite constellation name as string (e.g. `'OneWeb'`, `'STARLINK'`).
---

**2. Satellite Filtering & Properties (`satcon_properties`)**

This module enables the selection and visualiation of satellite datasets. Users can generate subsets of satellite populations in order to explore their collective metrics, analysing their individual trajectory properties, and visualise their orbital paths. 

**(i) Flyover Selection:**
- `get_norads(all_sat_data)`: Returns list of satellite NORAD IDs from the loaded dataset.
- `filter_by_norads(all_sat_data, exact_id=None, subset=None)`: Select satellites by NORAD IDs.
- `filter_by_time(all_sat_data, start_time, end_time)`: Select satellites visible in a time range (parameters are `datetime` objects).
- `filter_nth(all_sat_data, n=1)`: Keep every nth satellite from the dataset.
- `filter_custom(all_sat_data, custom_fn)`: Filter using a custom user-defined function.

**(ii) Satellite Properties & Plots:**

```python
from AstroTrack.satcon_properties import (
    plot_satellite_trajectory,
    plot_flyover_histogram_by_norad,
    plot_satellite_metric,
    plot_max_elevation_histogram
)
```

Function Signatures:

```python
plot_satellite_trajectory(
    satellite_data,
    elev: float=30,
    azim: float=45,
    time=None,
    ref_points=None,
    show_legend: bool=True,
    figsize=(8, 8),
    font_family="Times New Roman"
)

plot_flyover_histogram_by_norad(
    satellite_data,
    cutoff_norad=None,
    gap_hours: float=1.0,
    bin_width: int=500,
    color="lightpink",
    font_family="Times New Roman",
    figsize=(8, 8)
)

plot_satellite_metric(
    satellite_data,
    variable="Elevations",
    threshold=None,
    invert: bool=False,
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
- `satellite_data`: Output from `load_satellite_data()`.
- `time`: List of Skyfield time objects to compute satellite positions at.
- `ref_points`: Optional list of [label: str, latitude: float, longitude: float] reference locations (e.g. `["London", 51.5, 0.128]`).
- `elev`, `azim`, `figsize`, `font_family`, `color`, `bin_width`, `show_legend`: Optional plotting parameters.
- `variable`: Satellite metric selection (e.g. `"Elevations"`, `"Distances"`).
- Other thresholds or filtering parameters where applicable.

In order to initialise the epoch times, the following code can be used and manipulated as desired:

```python
import numpy as np
from skyfield.api import load

# Create a timescale object:
ts = load.timescale()

# Define orbit duration:
orbit_duration = np.arange(0, 3600, 1)  # in seconds

# Generate time array:
epochs_of_orbit = ts.utc(2025, 1, 1, 10, 15, 0 + orbit_duration)
```
---

**3. Doppler Analysis (`doppler_analysis`)**

This module computes and plots the Doppler shifts of satellites at given emission frequencies provided by the user, with respect to their observational ground-site. This assesses the detectability limits of a satellite's potential IEMR/UEMR, and visualises the evolution of flyovers.

```python
from AstroTrack.doppler_analysis import (
    plot_doppler_shifts, 
    check_doppler_resolution
)
```

Function Signatures:

```python
plot_doppler_shifts(
    satellite_data,
    f0,
    font_family="Times New Roman",
    figsize=(8, 5),
    marker_color="deeppink",
    time_window=10
)

check_doppler_resolution(
    all_satellite_data,
    f0_array,
    resolution=12_000,
    experiment="REACH",
    return_df: bool=False
)
```
**User Inputs**:
- `f0`, `f0_array`: Frequencies to compute satellite Doppler shifts in Hertz.
- `resolution`: Frequency resolution of experiment in Hertz.
- `time_window`: Half-width of visibility time over experiment in minutes.
- `marker_color`: Optional plotting parameter.
---

**4. Trajectory Animations (`satcon_animate`)**

This module creates three-dimensional animations of satellite flyovers projected on an Earth model, visualising it's orbital evolution with optional reference markers for ground-based locations of interest.

```python
from AstroTrack.satcon_animate import animate_trajectories
```
```python
animate_trajectories(
    satellite_data,
    elev: float=30,
    azim: float=300,
    ref_points=None,
    duration_hours: float=4,
    step_seconds: int=60,
    start_time=None,
    output_dir=".",
    filename_prefix="animated_satellites",
    font_family="Times New Roman"
)

```

**User Inputs**:
- `duration_hours`: Duration of satellite propagation in animation in hours.
- `step_seconds`: Time step between animation frames in seconds.
- `start_time`: Datetime object defining start of animation.
---

**5. Spectral Analysis with Satellite Metrics (`psd_analysis`)**

This module allows for the cross-correlation of radio spectra with satellite positions and other metrics, allowing for temporal relationships between RFI and the occurence of a satellite flyover.

```python
from AstroTrack.psd_analysis import (
    load_hdf5,
    plot_psd_with_satellite_metric,
    plot_psd_satellite_time_series
)

```

Function Signatures:

```python
load_hdf5(
    file_path: str,
    spectra_key: str="antenna_spectra",
    timestamps_key: str="antenna_timestamps"
)

plot_psd_with_satellite_metric(
    spectra: np.ndarray,
    timestamps: list[str],
    satellite_data: list[dict],
    variable: str="Elevations",
    bandwidth: float=200,
    freq_low_mhz: float=40,
    freq_high_mhz: float=170,
    v_min: float=1e14,
    v_max: float=8e16,
    show_legend: bool=False,
    threshold: float=None,
    vertical_lines: list[str]=None,
    cmap: str="Magma"
)

plot_psd_satellite_time_series(
    spectra: np.ndarray,
    timestamps: list[str],
    satellite_data: list[dict],
    norad_list: list[str]=None,
    satellite_variable: str="Elevations",
    R: float=2000,
    psd_freq_ranges: list[tuple]=None,
    target_freqs_mhz: list[float]=None,
    bandwidth: int=200,
    vmin: float=1e14,
    vmax: float=8e16,
    cmap: str="magma",
    line_colors: list[str]=None,
    threshold: float=None,
    vertical_lines: list[str]=None
)

```

**User Inputs**:
- `spectra`: Two-dimensional Power Spectral Density (PSD) array, with time x frequency.
- `timestamps`: UTC Timestamp strings per time bin.
- `bandwidth`: Experiment bandwidth in MHz.
- `freq_low_mhz`, `freq_high_mhz`: Chosen upper and lower frequency bounds for analysis in MHz.
- `psd_freq_ranges`: List of frequency ranges for multi-panel analysis such that [(low1, high1), (low2, high2), ...] in MHz.
- `norad_list`: List of NORAD ID's to only select specific satellites for plotting their metric.
- `threshold`: Variable threshold to hide satellite metrics beyond.
- `vmin`, `vmax`, `cmap`: Optional plotting parameters.

---

Example Workflow:
----------
Please refer to `test.py` in the `tests` directory for a more comprehensive workflow with **AstroTrack**, including usage of an example horizon profile and TLE catalogue.
