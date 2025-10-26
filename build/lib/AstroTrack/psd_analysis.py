from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
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
        spectra (np.ndarray): 2D array of PSD measurements (time x frequency bins)
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
                datetime.fromtimestamp(ts[0]).strftime("%H:%M:%S") for ts in timestamps
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
        (tuple[int]): Start and end bin index corresponding to freq_min and freq_max.
    """
    return hdf5_index(freq_min, total_bins, full_bandwidth_mhz), freq_index(
        freq_max, total_bins, full_bandwidth_mhz
    )


def plot_psd_with_satellite_metric(
    spectra: np.ndarray,
    utc_timestamps: list[str],
    all_satellite_data: list[dict],
    variable: str = "Elevations",
    freq_low_mhz: float = 40,
    freq_high_mhz: float = 170,
    v_min: float = 1e14,
    v_max: float = 8e16,
    show_legend: bool = False,
    threshold: float = None,
    vertical_lines: list[str] = None,
    cmap: str = "Magma",
):
    """
    Plot PSD waterfall and chosen satellite variable aligned by time.

    Parameters:
    spectra (2D np.ndarray): PSD measurements (time x frequency bins).
    utc_timestamps (list[str]): 'HH:MM:SS' formatted timestamps matching spectra.
    all_satellite_data (list of dict): Preprocessed satellite data.
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
        freq_low_mhz, freq_high_mhz, total_bins
    )
    spectra_subset = spectra[:, bin_start:bin_end]

    utc_to_idx = {t: i for i, t in enumerate(utc_timestamps)}
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
    ax1.set_title(f"Power Spectral Density ({freq_low_mhz}-{freq_high_mhz} MHz)")

    norads = []
    for sat_data in all_satellite_data:
        tle_line_2 = sat_data["TLE"][1]
        norad_id = tle_line_2.split()[1]
        values = np.array(sat_data[variable])
        times = [
            ep[-8:] if isinstance(ep, str) else ep.strftime("%H:%M:%S")
            for ep in sat_data["Epochs"]
        ]
        if threshold is not None:
            if variable == "Distances" and not np.any(values < threshold):
                continue
            if variable == "Elevations" and not np.any(values > threshold):
                continue

        aligned_vals = np.full(num_timestamps, np.nan)
        for t_str, val in zip(times, values):
            if t_str in utc_to_idx:
                aligned_vals[utc_to_idx[t_str]] = val

        if np.any(~np.isnan(aligned_vals)):
            ax2.plot(np.arange(num_timestamps), aligned_vals, label=norad_id, lw=1)
            norads.append(norad_id)

    if variable == "Distances":
        ax2.invert_yaxis()
    ax2.set_ylabel(variable)
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.set_title(f"Satellite {variable} Over Time")

    x_indices = np.linspace(0, num_timestamps - 1, min(30, num_timestamps), dtype=int)
    x_labels = [utc_timestamps[i] for i in x_indices]
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.set_xlim(0, num_timestamps - 1)
    ax2.set_xlim(0, num_timestamps - 1)

    if vertical_lines is not None:
        for t_str in vertical_lines:
            if t_str in utc_to_idx:
                idx = utc_to_idx[t_str]
                for ax in (ax1, ax2):
                    ax.axvline(idx, color="red", linestyle="--", lw=1)

    if show_legend:
        ax2.legend(loc="upper right", title="NORAD IDs")

    fig.suptitle(
        f"Flyovers from {utc_timestamps[0]} to {utc_timestamps[-1]} "
        f"({len(all_satellite_data)} satellites)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

    print(f"Plotted NORADs: {norads} | Count: {len(norads)}")


def plot_psd_satellite_time_series(
    spectra: np.ndarray,
    utc_timestamps: list[str],
    all_sat_data: list[dict],
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
    Plot PSD panels at single/multiple frequency, with corresponding time-series, and
    chosen satellite variable (Elevations, Distances, etc.) aligned by time.

    Parameters:
        spectra (np.ndarray): 2D array (time x frequency bins) of PSD measurements.
        utc_timestamps (list[str]): Time strings in 'HH:MM:SS' format.
        all_sat_data (list[dict]): Preprocessed satellite data.
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
    utc_to_idx = {t: i for i, t in enumerate(utc_timestamps)}

    if psd_freq_ranges is None:
        psd_freq_ranges = [(0, bandwidth)]  # default to full bandwidth

    num_psd_panels = len(psd_freq_ranges)
    num_sat_panels = len(norad_list)

    height_ratios = [1] + [2.5] * num_psd_panels + [0.6] * num_sat_panels
    fig, axes = plt.subplots(
        1 + num_psd_panels + num_sat_panels,
        1,
        figsize=(14, 4 + 1.1 * (num_psd_panels + num_sat_panels)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.15},
    )
    if len(axes.shape) == 0:
        axes = [axes]

    ax_time = axes[0]
    if target_freqs_mhz is not None:
        freq_bins = {}
        for f in target_freqs_mhz:
            bin_idx = int(round(f / bandwidth * total_bins))
            bin_idx = max(0, min(bin_idx, total_bins - 1))
            if bin_idx not in freq_bins:
                freq_bins[bin_idx] = [f]
            else:
                freq_bins[bin_idx].append(f)

        for b, freqs in freq_bins.items():
            ts = spectra[:, b]
            label = ", ".join(f"{freq:.1f} MHz" for freq in freqs)
            ax_time.plot(np.arange(num_timestamps), ts, lw=1.6, label=label)

        ax_time.set_ylabel("PSD")
        ax_time.set_title("Narrowband Frequency Time-Series")
        ax_time.legend(loc="upper right", fontsize=8, frameon=True)

    cax = fig.add_axes([0.92, 0.55, 0.02, 0.35]) if num_psd_panels > 0 else None
    for i, (low, high) in enumerate(psd_freq_ranges):
        ax = axes[1 + i]
        start_bin = int(round(low / bandwidth * total_bins))
        end_bin = int(round(high / bandwidth * total_bins))
        end_bin = min(end_bin, total_bins - 1)
        subset = spectra[:, start_bin:end_bin]
        im = ax.imshow(
            subset.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, num_timestamps - 1, low, high],
        )
        ax.set_ylabel("Freq (MHz)")
        ax.text(
            0.01,
            0.95,
            f"{low}-{high} MHz",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="maroon",
        )
        if i == 0 and cax is not None:
            fig.colorbar(im, cax=cax, label="PSD Intensity")

    sat_dict = {sat["TLE"][1].split()[1]: sat for sat in all_sat_data}
    for i, norad in enumerate(norad_list):
        ax = axes[1 + num_psd_panels + i]
        if norad not in sat_dict:
            ax.text(
                0.5,
                0.5,
                f"No data for NORAD {norad}",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            continue

        sat = sat_dict[norad]
        values = np.array(sat[satellite_variable])
        times = [
            ep[-8:] if isinstance(ep, str) else ep.strftime("%H:%M:%S")
            for ep in sat["Epochs"]
        ]
        aligned = np.full(num_timestamps, np.nan)
        for t_str, val in zip(times, values):
            if t_str in utc_to_idx:
                aligned[utc_to_idx[t_str]] = val

        if threshold is not None:
            if satellite_variable == "Distances" and not np.any(values < threshold):
                ax.set_visible(False)
                continue
            if satellite_variable == "Elevations" and not np.any(values > threshold):
                ax.set_visible(False)
                continue

        ax.plot(np.arange(num_timestamps), aligned, lw=1, color="deeppink")
        ax.text(
            1.01,
            0.5,
            f"NORAD {norad}",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9,
        )

        valid = aligned[~np.isnan(aligned)]
        if len(valid) > 0:
            margin = (
                0.05 * (valid.max() - valid.min()) if valid.max() != valid.min() else 1
            )
            ax.set_ylim(valid.min() - margin, valid.max() + margin)
        if satellite_variable == "Distances":
            ax.set_ylim(0, R)
            ax.invert_yaxis()
        elif satellite_variable == "Elevations":
            ax.set_ylim(0, 90)

    x_indices = np.linspace(0, num_timestamps - 1, min(30, num_timestamps), dtype=int)
    x_labels = [utc_timestamps[i] for i in x_indices]
    axes[-1].set_xticks(x_indices)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Timestamp (UTC)")

    if vertical_lines:
        for t_str in vertical_lines:
            idx = utc_to_idx.get(t_str, None)
            if idx is not None:
                for ax in axes:
                    ax.axvline(idx, color="red", linestyle="--", lw=1)

    if satellite_variable == "Distances":
        fig.text(
            0.08, 0.3, "Distance (km)", va="center", rotation="vertical", fontsize=10
        )
    else:
        fig.text(
            0.08,
            0.3,
            r"Elevation ($\degree$)",
            va="center",
            rotation="vertical",
            fontsize=10,
        )

    fig.suptitle(f"PSD and Satellite {satellite_variable}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()
