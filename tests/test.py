# run_pipeline.py
from datetime import datetime
from AstroTrack.preprocess import load_satellite_data
from AstroTrack.satcon_properties import (
    plot_satellite_trajectory,
    plot_flyover_histogram_by_norad,
    plot_satellite_metric,
    filter_by_norads
)
from AstroTrack.doppler_analysis import check_doppler_resolution, plot_doppler_shifts
import matplotlib.pyplot as plt

data = load_satellite_data(
    tle_file="LEO_TLE_file_14_06_2025_time_17_10.txt",
    target_date=datetime(2025,6,15,12,25,8),
    obs_len=3600,
    traj_res=60,
    obs_lat=51.5,
    obs_lon=-0.1,
    R=2000,
    horizon_data="REACH-Horizon.csv",
    satcon="OneWeb"
)

plot_satellite_trajectory(
    data,
    elev=30,
    azim=45,
    time=None,
    ref_points=None,
    show_legend=True,
    figsize=(8, 8),
    font_family="Times New Roman"
)

plot_flyover_histogram_by_norad(
    data,
    cutoff_norad=None,
    gap_hours=1.0,
    bin_width=500,
    color="lightpink",
    font_family="Times New Roman",
    figsize=(8, 8)
)

plot_satellite_metric(
    data,
    variable="Elevations",
    threshold=None,
    invert=False,
    font_family="Times New Roman",
    figsize=(8, 8)
)

subset = filter_by_norads(data, exact_id=56713)
f0_array=[110e6, 137.5e6]
check_doppler_resolution(subset, f0_array)
plot_doppler_shifts(data,f0_array[0])

