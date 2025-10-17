from skyfield.api import EarthSatellite, load
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


def lat_lon_to_xyz(re, lat, lon):
    """Convert latitude and longitude (degrees) to XYZ coordinates (km)."""
    x = re * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = re * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = re * np.sin(np.radians(lat))
    return x, y, z


def timestamps(ts_array):
    """Convert Skyfield times to UTC string list."""
    return [t.utc_strftime("%Y-%m-%d %H:%M:%S") for t in ts_array]


def update_frame(frame, trajectories, plots, timestamps, timestamp_text):
    """Frame update function for FuncAnimation."""
    for i, scatter in enumerate(plots):
        pos = trajectories[i]
        scatter._offsets3d = ([pos[0, frame]], [pos[1, frame]], [pos[2, frame]])
    timestamp_text.set_text(timestamps[frame])
    return plots


def animate_trajectories(
    all_satellite_data,
    elev: float = 30,
    azim: float = 300,
    ref_points=None,
    duration_hours: float = 4,
    step_seconds: int = 60,
    start_time=None,
    output_dir=".",
    filename_prefix="animated_satellites",
    font_family="Times New Roman",
):
    """
    Animate 3D satellite trajectories around Earth and save as GIF.

    Parameters:
        all_satellite_data (list[dict]): Satellite data dictionaries (from preprocess module).
        elev, azim (float): 3D view angles for the animation.
        ref_points (list[tuple[str, float, float]]): Optional list of (label, lat, lon) reference locations.
        duration_hours (float): Duration of animation window.
        step_seconds (int): Time step between frames.
        start_time (datetime): Start time of animation.
        output_dir (str): Directory to save animation file.
        filename_prefix (str): Prefix for output file name.
        font_family (str): Font family for plot text.
    """
    plt.rcParams["font.family"] = font_family

    ts = load.timescale()
    re = 6378.0  # Earth radius (km)
    theta = np.linspace(0, 2 * np.pi, 201)
    cth, sth, zth = np.cos(theta), np.sin(theta), np.zeros_like(theta)
    lon_lines, lat_lines = [], []

    for phi in np.radians(np.arange(0, 180, 15)):
        cph, sph = np.cos(phi), np.sin(phi)
        lon_lines.append(
            (re * cth * cph, re * sth * cph, re * sph * np.ones_like(theta))
        )

    for phi in np.radians(np.arange(-75, 90, 15)):
        cph, sph = np.cos(phi), np.sin(phi)
        lat_lines.append(
            (
                re * np.cos(theta) * cph,
                re * np.sin(theta) * cph,
                re * np.ones_like(theta) * sph,
            )
        )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    for x, y, z in lon_lines + lat_lines:
        ax.plot(x, y, z, "-k", lw=0.5, alpha=0.4)

    if ref_points:
        for label, lat, lon in ref_points:
            x, y, z = lat_lon_to_xyz(re, lat, lon)
            ax.scatter(x, y, z, s=80, alpha=0.7, label=label)

    plots, trajectories = [], []
    if start_time is not None:
        if isinstance(start_time, datetime):
            start_dt = start_time
        else:
            raise TypeError("start_time must be a datetime.datetime object.")
    else:
        first_sat = all_satellite_data[0]
        start_dt = datetime.fromisoformat(str(first_sat["Epochs"][0]))

    orbit_duration = np.arange(0, duration_hours * 3600, step_seconds)
    time_of_orbit = ts.utc(
        start_dt.year,
        start_dt.month,
        start_dt.day,
        start_dt.hour,
        start_dt.minute,
        start_dt.second + orbit_duration,
    )

    for sat_data in all_satellite_data:
        L1, L2 = sat_data["TLE"]
        sat = EarthSatellite(L1, L2, ts=ts)
        pos = sat.at(time_of_orbit).position.km
        scatter = ax.scatter(
            [pos[0, 0]], [pos[1, 0]], [pos[2, 0]], label=L2.split()[1], s=15
        )
        plots.append(scatter)
        trajectories.append(pos)

    timestamps = timestamps(time_of_orbit)
    timestamp_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)

    ax.view_init(elev, azim)
    ax.set_title(f"Satellite Animation (elev={elev}°, azim={azim}°)")
    ax.legend(loc="upper right")

    ani = FuncAnimation(
        fig,
        update_frame,
        frames=len(timestamps),
        fargs=(trajectories, plots, timestamps, ax, timestamp_text),
        interval=100,
        blit=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename_prefix}_e{elev}_a{azim}.gif")
    ani.save(filepath, dpi=80)
    plt.close()
    print(f"Animation saved: {filepath}")
