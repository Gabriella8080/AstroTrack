# run_pipeline.py
from datetime import datetime
from AstroTrack.preprocess import load_satellite_data
from AstroTrack.satcon_properties import filter_by_norads
from AstroTrack.doppler_analysis import check_doppler_resolution

data = load_satellite_data(
    tle_file="LEO_TLE_file_14_06_2025_time_17_10.txt",
    target_date=datetime(2025,10,12,12,0,0),
    obs_len=3600,
    traj_res=60,
    obs_lat=51.5,
    obs_lon=-0.1,
    R=2000,
    horizon_data="REACH-Horizon.csv",
    satcon="OneWeb"
)

subset = filter_by_norads(data, exact_id=63115)
check_doppler_resolution(subset, f0_array=[110e6, 137.5e6])

