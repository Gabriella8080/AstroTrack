from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams["font.family"] = "Times New Roman"


def load_hdf5(
    file_path: str,
    spectra_key: str = "ant_spectra",
    timestamps_key: str = "ant_timestamps",
):
    """
    Load observation HDF5 file to return spectra and UTC timestamps.

    Parameters:
        file_path (str): Path to the user's HDF5 file.
        spectra_key (str): Dataset name for spectra.
        timestamps_key (str): Dataset name for timestamps.

    Outputs:
        spectra (np.ndarray): 2D array of PSD measurements
                            (time x frequency bins)
        utc_timestamps (list[str]): List of 'HH:MM:SS' formatted UTC timestamps
    """
    try:
        with h5py.File(file_path, "r") as f:
            if "observation_data" not in f:
                raise ValueError("HDF5 file missing 'observation_data' group")

            obs_group = f["observation_data"]
            if spectra_key not in obs_group:
                raise ValueError(f"HDF5 file missing spectra dataset: '{spectra_key}'")
            if timestamps_key not in obs_group:
                raise ValueError(
                    f"HDF5 file missing timestamps dataset: '{timestamps_key}'"
                )

            spectra = obs_group[spectra_key][:]
            timestamps = obs_group[timestamps_key][:]
            utc_timestamps = [
                datetime.fromtimestamp(ts[0]).strftime("%H:%M:%S")
                for ts in timestamps  # noqa: E501
            ]

    except (OSError, KeyError, ValueError) as e:
        raise RuntimeError(f"Error reading HDF5 file: {e}")

    return spectra, utc_timestamps


def hdf5_index(freq_mhz, total_bins, full_bandwidth_mhz=200):
    """Convert frequency (MHz) to bin index for spectral data.

    Parameters:
        freq_mhz (float): Single frequency [MHz].
        total_bins (int): Total number of bins in spectral dataset.
        full_bandwidth_mhz (float): Total bandwidth of spectrum [MHz].

    Output:
        (int): Corresponding bin index.
    """
    bin_idx = int(round(freq_mhz / (full_bandwidth_mhz / total_bins)))
    return max(0, min(bin_idx, total_bins - 1))


def freq_index(bin_idx, total_bins, full_bandwidth_mhz=200):
    """Convert bin index to true frequency (MHz) of spectral data.

    Parameters:
        bin_idx (int): Bin index to convert.
        total_bins (int): Total number of bins in spectral dataset.
        full_bandwidth_mhz (float): Total bandwidth of spectrum [MHz].

    Output:
        (float): Frequency corresponding to bin index [MHz].
    """
    return bin_idx * (full_bandwidth_mhz / total_bins)


def get_frequency_bin_range(freq_min, freq_max, total_bins, full_bandwidth_mhz=200):
    """
    Compute start and end bin indices for specified frequency range.

    Parameters:
        freq_min (float): Minimum frequency of range [MHz].
        freq_max (float): Maximum frequency of range [MHz].
        total_bins (int): Total number of bins in spectral dataset.
        full_bandwidth_mhz (float): Total bandwidth of spectrum [MHz].

    Output:
        (tuple[int]): Start and end bin index corresponding
        to freq_min and freq_max.
    """
    return (
        hdf5_index(freq_min, total_bins, full_bandwidth_mhz),
        hdf5_index(freq_max, total_bins, full_bandwidth_mhz),
    )


def iso_to_hms(t):
    """
    Convert datetime/ISO string/timestamp from HDF5 file
    used by REACH collaboration to 'HH:MM:SS' format.
    """
    if isinstance(t, str):
        t = t.rstrip("Z")
        if "T" in t:
            t = t.split("T")[-1]
        return t[:8]

    if isinstance(t, datetime):
        return t.strftime("%H:%M:%S")

    raise TypeError(f"Unsupported time format: {type(t)}")


def build_time_index_map(utc_timestamps):
    """
    Map UTC timeline to indices using normalised HH:MM:SS keys.
    """
    return {iso_to_hms(t): i for i, t in enumerate(utc_timestamps)}


def plot_psd_with_satellite_metric(
    spectra: np.ndarray,
    utc_timestamps: list[str],
    satellite_data: list[dict],
    variable: str = "Elevations",
    bandwidth: float = 200,
    freq_low_mhz: float = 40,
    freq_high_mhz: float = 170,
    v_min: float = 1e14,
    v_max: float = 8e16,
    show_legend: bool = False,
    threshold: float = None,
    vertical_lines: list[str] = None,
    cmap: str = "magma",
):
    """
    Plot PSD waterfall and chosen satellite variable aligned by time.

    Parameters:
    spectra (2D np.ndarray): PSD measurements (time x frequency bins).
    utc_timestamps (list[str]): 'HH:MM:SS' formatted
                                timestamps matching spectra.
    satellite_data (list of dict): Preprocessed satellite data.
    variable (str): Satellite variable to plot ('Elevations', 'Distances').
    freq_low_mhz, freq_high_mhz (float): Frequency range to plot [MHz].
    v_min, v_max (float): Color scale for PSD.
    show_legend (bool): Display NORAD IDs in plot.
    threshold (float): Filter satellites by variable threshold.
    vertical_lines (list of str): Timestamp lines to mark.
    cmap (str): Colormap for PSD.
    """
    num_timestamps, total_bins = spectra.shape
    bin_start, bin_end = get_frequency_bin_range(
        freq_low_mhz, freq_high_mhz, total_bins, bandwidth
    )
    bin_start, bin_end = int(bin_start), int(bin_end)
    spectra_subset = spectra[:, bin_start:bin_end]
    utc_to_idx = build_time_index_map(utc_timestamps)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [4, 2]}
    )

    cax = fig.add_axes([0.92, 0.105, 0.02, 0.775])

    im = ax1.imshow(
        spectra_subset.T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=v_min,
        vmax=v_max,
        extent=[0, num_timestamps - 1, freq_low_mhz, freq_high_mhz],
    )

    fig.colorbar(im, cax=cax, label="PSD Intensity")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_title(f"PSD ({freq_low_mhz}-{freq_high_mhz} MHz)")

    norads = []
    for sat_data in satellite_data:
        if "TLE" not in sat_data:
            continue
        norad_id = sat_data["TLE"][1].split()[1]
        values = np.array(sat_data[variable])
        times = [iso_to_hms(ep) for ep in sat_data["Epochs"]]
        if threshold is not None:
            if variable == "Distances" and not np.any(values < threshold):
                continue
            if variable == "Elevations" and not np.any(values > threshold):
                continue

        aligned = np.full(num_timestamps, np.nan)
        matched = 0
        for t, v in zip(times, values):
            idx = utc_to_idx.get(t)
            if idx is not None:
                aligned[idx] = v
                matched += 1
        if matched == 0:
            continue
        ax2.plot(np.arange(num_timestamps), aligned, lw=1, label=norad_id)
        norads.append(norad_id)

    if len(norads) == 0:
        print("No satellites flyovers matched HDF5 Observation UTC timeline.")
        return

    if variable == "Distances":
        ax2.invert_yaxis()
    ax2.set_ylabel(variable)
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.set_title(f"Satellite {variable}")
    x_idx = np.linspace(0, num_timestamps - 1, min(30, num_timestamps), dtype=int)
    x_lbl = [utc_timestamps[i] for i in x_idx]
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_lbl, rotation=45, ha="right")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_lbl, rotation=45, ha="right")

    if vertical_lines:
        for t in vertical_lines:
            idx = utc_to_idx.get(iso_to_hms(t))
            if idx is not None:
                ax1.axvline(idx, color="red", ls="--", lw=1)
                ax2.axvline(idx, color="red", ls="--", lw=1)

    if show_legend:
        ax2.legend()

    fig.suptitle(
        f"Flyovers ({len(norads)} satellites matched)",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def plot_psd_satellite_time_series(
    spectra: np.ndarray,
    utc_timestamps: list[str],
    satellite_data: list[dict],
    norad_list: list[str] = None,
    satellite_variable: str = "Elevations",
    R: float = 2000,
    psd_freq_ranges: list[tuple] = None,
    target_freqs_mhz: list[float] = None,
    bandwidth: int = 200,
    vmin: float = 1e14,
    vmax: float = 8e16,
    cmap: str = "magma",
    line_colors: list[str] = None,
    threshold: float = None,
    vertical_lines: list[str] = None,
):
    """
    Plot PSD panels at single/multiple frequency, with corresponding
    time-series, and chosen satellite variable (Elevations, Distances, etc.) aligned by time.

    Parameters:
        spectra (np.ndarray): 2D array (time x frequency bins) of PSD measurements.
        utc_timestamps (list[str]): Time strings in 'HH:MM:SS' format.
        satellite_data (list[dict]): Preprocessed satellite data.
        norad_list (list[str]): List of NORAD IDs for satellite panels, if provided.
        satellite_variable (str): Satellite variable to plot ('Elevations' or 'Distances').
        R (float): Radial constraint [km].
        psd_freq_ranges (list of tuples): [(low1, high1), (low2, high2), ...]; if None, full range used [MHz].
        target_freqs_mhz (list of floats): Frequencies to plot as narrowband time-series; if None, skip [MHz].
        bandwidth (int): Frequency bandwidth of spectral dataset [MHz].
        vmin, vmax (float): Color scale limits for PSD.
        cmap (str): Colormap for PSD.
        line_colors (list of str): Colors for narrowband time-series,
        threshold (float): Optional threshold to hide satellites.
        vertical_lines (list[str]): UTC times to draw vertical dashed lines.
    """
    num_timestamps, total_bins = spectra.shape
    utc_to_idx = build_time_index_map(utc_timestamps)
    if psd_freq_ranges is None:
        psd_freq_ranges = [(0, bandwidth)]

    if norad_list is None:
        norad_list = [
            sat["TLE"][1].split()[1] for sat in satellite_data if "TLE" in sat
        ]

    if len(norad_list) == 0:
        print("No NORAD list made available.")
        return

    num_psd = len(psd_freq_ranges)
    num_sat = len(norad_list)
    fig, axes = plt.subplots(
        1 + num_psd + num_sat,
        1,
        figsize=(14, 4 + 1.1 * (num_psd + num_sat)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    # PSD panel(s)
    for i, (low, high) in enumerate(psd_freq_ranges):
        ax = axes[1 + i]
        lower = int(round(low / bandwidth * total_bins))
        upper = int(round(high / bandwidth * total_bins))
        upper = min(upper, total_bins - 1)
        im = ax.imshow(
            spectra[:, lower:upper].T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_ylabel("Freq (MHz)")

    # Satellite panel(s)
    sat_dict = {sat["TLE"][1].split()[1]: sat for sat in satellite_data if "TLE" in sat}
    plotted = []
    for i, norad in enumerate(norad_list):
        ax = axes[1 + num_psd + i]

        if norad not in sat_dict:
            ax.text(0.5, 0.5, f"No data {norad}", ha="center")
            continue

        sat = sat_dict[norad]
        values = np.array(sat[satellite_variable])
        times = [iso_to_hms(ep) for ep in sat["Epochs"]]
        aligned = np.full(num_timestamps, np.nan)
        matched = 0
        for t, v in zip(times, values):
            idx = utc_to_idx.get(t)
            if idx is not None:
                aligned[idx] = v
                matched += 1

        if matched == 0:
            continue

        ax.plot(aligned, lw=1)
        plotted.append(norad)
        if satellite_variable == "Distances":
            ax.set_ylim(0, R)
            ax.invert_yaxis()
        elif satellite_variable == "Elevations":
            ax.set_ylim(0, 90)

    axes[-1].set_xlabel("Timestamp (UTC)")

    if len(plotted) == 0:
        print("No satellites flyovers matched HDF5 Observation UTC timeline.")
        return

    plt.tight_layout()
    plt.show()

