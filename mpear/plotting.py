import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
from astropy import units as u
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable

import astropy.constants as const
from .orbital_mechanics import *
from .flux_calculations import *

import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.patheffects as PathEffects
from matplotlib.legend_handler import HandlerBase


class GradientLineHandler(HandlerBase):
    def __init__(self, cmap):
        self.cmap = cmap
        super().__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Create a gradient array
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        grad = np.vstack((grad, grad))

        # Create an image of the gradient
        gradient_image = self.cmap(grad)

        # Create a Rectangle patch to hold the gradient image
        rect = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor="none",
            edgecolor="none",
            transform=trans,
        )

        # Create an AxesImage for the gradient
        ax = legend.axes
        img = ax.imshow(
            gradient_image,
            aspect="auto",
            extent=[xdescent, xdescent + width, ydescent, ydescent + height],
            clip_on=True,
        )

        # Set the clip path to be the Rectangle
        img.set_clip_path(rect)

        return [rect, img]


# Font sizes for full-width figures
FULL_WIDTH_SIZES = {
    "TITLE_SIZE": 24,
    "SUBTITLE_SIZE": 18,
    "LABEL_SIZE": 18,
    "TICK_SIZE": 14,
    "LEGEND_SIZE": 12,
}

# Font sizes for half-width figures
HALF_WIDTH_SIZES = {
    "TITLE_SIZE": 24,
    "SUBTITLE_SIZE": 22,
    "LABEL_SIZE": 20,
    "TICK_SIZE": 16,
    "LEGEND_SIZE": 16,
}


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def get_colormaps():
    cdict = {
        "red": (
            (0.0, inter_from_256(0), inter_from_256(0)),  # Rich black
            (1 / 4 * 1, inter_from_256(0), inter_from_256(0)),  # Midnight Green
            (1 / 4 * 2, inter_from_256(10), inter_from_256(10)),  # Dark Cyan
            (1 / 4 * 3, inter_from_256(148), inter_from_256(148)),  # Tiffany Blue
            (1 / 4 * 4, inter_from_256(233), inter_from_256(233)),
        ),  # Vanilla
        "green": (
            (0.0, inter_from_256(18), inter_from_256(18)),  # Rich Black
            (1 / 4 * 1, inter_from_256(95), inter_from_256(95)),  # Midnight Green
            (1 / 4 * 2, inter_from_256(147), inter_from_256(147)),  # Dark Cyan
            (1 / 4 * 3, inter_from_256(210), inter_from_256(210)),  # Tiffany Blue
            (1 / 4 * 4, inter_from_256(216), inter_from_256(216)),
        ),  # Vanilla
        "blue": (
            (0.0, inter_from_256(25), inter_from_256(25)),  # Rich Black
            (1 / 4 * 1, inter_from_256(115), inter_from_256(115)),  # Midnight Green
            (1 / 4 * 2, inter_from_256(150), inter_from_256(150)),  # Dark Cyan
            (1 / 4 * 3, inter_from_256(189), inter_from_256(189)),  # Tiffany Blue
            (1 / 4 * 4, inter_from_256(166), inter_from_256(166)),
        ),  # Vanilla
    }
    cool_colormap = colors.LinearSegmentedColormap("cool_colormap", segmentdata=cdict)
    # fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')
    # fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(),cmap=cool_colormap),orientation='horizontal',cax=ax)

    cdict = {
        "red": (
            (0.0, inter_from_256(238), inter_from_256(238)),  # Gamboge
            (1 / 4 * 1, inter_from_256(202), inter_from_256(202)),  # Alloy Orange
            (1 / 4 * 2, inter_from_256(187), inter_from_256(187)),  # Rust
            (1 / 4 * 3, inter_from_256(174), inter_from_256(174)),  # Rufous
            (1 / 4 * 4, inter_from_256(155), inter_from_256(155)),
        ),  # Auburn
        "green": (
            (0.0, inter_from_256(155), inter_from_256(155)),  # Gamboge
            (1 / 4 * 1, inter_from_256(103), inter_from_256(104)),  # Alloy Orange
            (1 / 4 * 2, inter_from_256(62), inter_from_256(62)),  # Rust
            (1 / 4 * 3, inter_from_256(32), inter_from_256(32)),  # Rufous
            (1 / 4 * 4, inter_from_256(34), inter_from_256(34)),
        ),  # Auburn
        "blue": (
            (0.0, inter_from_256(0), inter_from_256(0)),  # Gamboge
            (1 / 4 * 1, inter_from_256(2), inter_from_256(2)),  # Alloy Orange
            (1 / 4 * 2, inter_from_256(3), inter_from_256(3)),  # Rust
            (1 / 4 * 3, inter_from_256(18), inter_from_256(18)),  # Rufous
            (1 / 4 * 4, inter_from_256(38), inter_from_256(38)),
        ),  # Auburn
    }
    warm_colormap = colors.LinearSegmentedColormap("warm_colormap", segmentdata=cdict)
    # fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')
    # fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(),cmap=warm_colormap),orientation='horizontal',cax=ax)

    cdict = {
        "red": (
            (0.0, inter_from_256(0), inter_from_256(0)),  # Rich black
            (1 / 8 * 1, inter_from_256(0), inter_from_256(0)),  # Midnight Green
            (1 / 8 * 2, inter_from_256(10), inter_from_256(10)),  # Dark Cyan
            (1 / 8 * 3, inter_from_256(148), inter_from_256(148)),  # Tiffany Blue
            (1 / 8 * 4, inter_from_256(233), inter_from_256(233)),  # Vanilla
            (1 / 8 * 5, inter_from_256(238), inter_from_256(238)),  # Gamboge
            (1 / 8 * 6, inter_from_256(202), inter_from_256(202)),  # Alloy Orange
            (1 / 8 * 7, inter_from_256(187), inter_from_256(187)),  # Rust
            #    (1/9*8,inter_from_256(174),inter_from_256(174)), # Rufous
            (1, inter_from_256(155), inter_from_256(155)),
        ),  # Auburn
        "green": (
            (0.0, inter_from_256(18), inter_from_256(18)),  # Rich Black
            (1 / 8 * 1, inter_from_256(95), inter_from_256(95)),  # Midnight Green
            (1 / 8 * 2, inter_from_256(147), inter_from_256(147)),  # Dark Cyan
            (1 / 8 * 3, inter_from_256(210), inter_from_256(210)),  # Tiffany Blue
            (1 / 8 * 4, inter_from_256(216), inter_from_256(216)),  # Vanilla
            (1 / 8 * 5, inter_from_256(155), inter_from_256(155)),  # Gamboge
            (1 / 8 * 6, inter_from_256(103), inter_from_256(104)),  # Alloy Orange
            (1 / 8 * 7, inter_from_256(62), inter_from_256(62)),  # Rust
            #     (1 / 9 * 8, inter_from_256(32), inter_from_256(32)), # Rufous
            (1, inter_from_256(34), inter_from_256(34)),
        ),  # Auburn
        "blue": (
            (0.0, inter_from_256(25), inter_from_256(25)),  # Rich Black
            (1 / 8 * 1, inter_from_256(115), inter_from_256(115)),  # Midnight Green
            (1 / 8 * 2, inter_from_256(150), inter_from_256(150)),  # Dark Cyan
            (1 / 8 * 3, inter_from_256(189), inter_from_256(189)),  # Tiffany Blue
            (1 / 8 * 4, inter_from_256(166), inter_from_256(166)),  # Vanilla
            (1 / 8 * 5, inter_from_256(0), inter_from_256(0)),  # Gamboge
            (1 / 8 * 6, inter_from_256(2), inter_from_256(2)),  # Alloy Orange
            (1 / 8 * 7, inter_from_256(3), inter_from_256(3)),  # Rust
            #       (1 / 9 * 8, inter_from_256(18), inter_from_256(18)), # Rufous
            (1, inter_from_256(38), inter_from_256(38)),
        ),  # Auburn
    }
    full_colormap = colors.LinearSegmentedColormap("full_colormap", segmentdata=cdict)
    # fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')
    # fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(),cmap=full_colormap),orientation='horizontal',cax=ax)

    return cool_colormap, warm_colormap, full_colormap


# Plotting functions
def plot_3d_scatter(
    matrix,
    tolerance,
    a_range,
    i_range,
    t_range,
    title,
    specific_points,
    specific_labels,
):

    cold_colormap, warm_colormap, full_colormap = get_colormaps()
    plt.rcParams.update(
        {"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]}
    )  # Set a base font size

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.where(matrix < tolerance)
    scatter = ax.scatter(
        a_range[z],
        i_range[x],
        t_range[y],
        c=matrix[x, y, z],
        cmap=full_colormap,
        zorder=-1,
    )

    for idx, point in enumerate(specific_points):
        ax.scatter(
            point[0], point[1], point[2], color="black", s=20, marker="s", zorder=10
        )

        # Add label
        offset = 2  # Adjust this value to control the distance between point and label
        ax.text(
            point[0] + offset,
            point[1] + offset,
            point[2] + offset,
            specific_labels[idx],
            color="black",
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=10,
        )

    ax.set_xlabel(r"$a$ (AU)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax.set_ylabel(r"$\bar{i}$ (degrees)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax.set_zlabel(r"$f$ (degrees)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax.set_title(title, fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
    plt.colorbar(scatter, label="Difference from target (arcsec)")
    plt.tight_layout()
    return fig


def plot_4d_scatter(matrix, tolerance, a_range, i_range, t_range, UpOmega_range, title):
    plt.rcParams.update(
        {"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]}
    )  # Set a base font size

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z, om = np.where(matrix < tolerance)
    scatter = ax.scatter(
        a_range[z],
        i_range[x],
        t_range[y],
        s=UpOmega_range[om] / 36,
        c=matrix[x, y, z, om],
        cmap="RdYlGn_r",
    )
    ax.set_xlabel(r"$a$ (AU)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax.set_ylabel(r"$\bar{i}$ (degrees)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax.set_zlabel(r"$f$ (degrees)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax.set_title(title, fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
    plt.colorbar(scatter, label="Difference from target (arcsec)")
    plt.tight_layout()
    return fig


def rotate_90_counterclockwise(x, y):
    return -y, x


def create_moon_markers():
    # 100%
    full_moon = Path.circle()

    # 0%
    new_moon = Path.circle()

    # 50%
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    x_rot, y_rot = rotate_90_counterclockwise(x, y)
    half_moon_verts = [(1, 0)] + list(zip(x_rot, y_rot)) + [(-1, 0)]
    half_moon = Path(half_moon_verts)

    # 10%
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = 0.7 * np.cos(theta)
    y2 = np.sin(theta)
    x1_rot, y1_rot = rotate_90_counterclockwise(x1, y1)
    x2_rot, y2_rot = rotate_90_counterclockwise(x2, y2)
    small_crescent_verts = list(zip(x1_rot, y1_rot)) + list(zip(x2_rot, y2_rot))[::-1]
    small_crescent = Path(small_crescent_verts)

    # 30%
    x2 = 0.3 * np.cos(theta)
    y2 = np.sin(theta)
    x2_rot, y2_rot = rotate_90_counterclockwise(x2, y2)
    large_crescent_verts = list(zip(x1_rot, y1_rot)) + list(zip(x2_rot, y2_rot))[::-1]
    large_crescent = Path(large_crescent_verts)

    # 90%
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = -0.7 * np.cos(theta)
    y2 = np.sin(theta)
    x1_rot, y1_rot = rotate_90_counterclockwise(x1, y1)
    x2_rot, y2_rot = rotate_90_counterclockwise(x2, y2)
    large_gibbous_verts = list(zip(x1_rot, y1_rot)) + list(zip(x2_rot, y2_rot))[::-1]
    large_gibbous = Path(large_gibbous_verts)

    # 70%
    x2 = -0.3 * np.cos(theta)
    y2 = np.sin(theta)
    x2_rot, y2_rot = rotate_90_counterclockwise(x2, y2)
    small_gibbous_verts = list(zip(x1_rot, y1_rot)) + list(zip(x2_rot, y2_rot))[::-1]
    small_gibbous = Path(small_gibbous_verts)

    return {
        "100%": MarkerStyle(full_moon),
        "0%": MarkerStyle(new_moon),
        "50%": MarkerStyle(half_moon),
        "10%": MarkerStyle(small_crescent),
        "30%": MarkerStyle(large_crescent),
        "70%": MarkerStyle(small_gibbous),
        "90%": MarkerStyle(large_gibbous),
    }


# # Create a plot to demonstrate the markers
# fig, ax = plt.subplots(figsize=(8, 4))  # Slightly reduced height
# moon_markers = create_moon_markers()
# phases = ["0%", "10%", "30%", "50%", "70%", "90%", "100%"]
# labels = [
#     r"$\phi\geq175^\circ$",
#     r"$150^\circ\leq\phi<175^\circ$",
#     r"$100^\circ\leq\phi<150^\circ$",
#     r"$80^\circ\leq\phi<100^\circ$",
#     r"$30^\circ\leq\phi<80^\circ$",
#     r"$5^\circ\leq\phi<30^\circ$",
#     r"$\phi<5^\circ$",
# ]

# num_rows = (len(phases) + 1) // 2  # Calculate number of rows needed
# y_spacing = 0.2  # Reduced spacing between rows

# for i, phase in enumerate(phases):
#     row = i % num_rows
#     col = i // num_rows

#     x = col * 4  # Spacing between columns
#     y = (num_rows - 1 - row) * y_spacing  # Closer spacing in y-axis

#     # Plot a black scatter point below the marker
#     ax.scatter(x, y, s=300, c="black", marker="o")  # Slightly reduced marker size

#     # Plot the moon phase marker
#     if phase == "0%":
#         ax.scatter(
#             x,
#             y,
#             s=300,
#             marker=moon_markers[phase],
#             facecolor="black",
#             edgecolor="black",
#         )
#     else:
#         ax.scatter(
#             x,
#             y,
#             s=300,
#             marker=moon_markers[phase],
#             facecolor="yellow",
#             edgecolor="black",
#         )

#     # Add the phase label
#     ax.text(
#         x + 0.5,
#         y,
#         labels[i],
#         va="center",
#         fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"],
#         ha="left",
#     )

# ax.set_xlim(-0.5, 6)
# ax.set_ylim(-0.2, (num_rows - 1) * y_spacing + 0.2)
# ax.axis("off")
# ax.set_title("Legend", fontsize=HALF_WIDTH_SIZES["TITLE_SIZE"])
# plt.tight_layout()  # Reduced padding
# fig.savefig("imgs/marker_legend.pdf", bbox_inches="tight")  # Reduced padding


def plot_specific_orbit(point, label, lims, UpOmega, omega, d):
    plt.rcParams.update(
        {"font.size": HALF_WIDTH_SIZES["TICK_SIZE"]}
    )  # Set a base font size

    # Create plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    moon_markers = create_moon_markers()

    a, i, t = point

    # Calculate full orbit
    orbit = calculate_full_orbit(
        r=a * const.au, inclination=np.radians(i), UpOmega=UpOmega, omega=omega, d=d
    )
    # Calculate specific point
    x, y, z = calculate_orbit_location(
        r=a * const.au,
        inclination=np.radians(i),
        true_anomaly=np.radians(t),
        UpOmega=UpOmega,
        omega=omega,
    )

    # Calculate phase angle
    phase_angle = calculate_apparent_phase(x, y, z, d)
    color = "gold"
    # Determine which marker to use based on the phase angle
    if phase_angle < 5:
        marker = moon_markers["100%"]
    elif 5 <= phase_angle < 30:
        marker = moon_markers["90%"]
    elif 30 <= phase_angle < 80:
        marker = moon_markers["70%"]
    elif 80 <= phase_angle < 100:
        marker = moon_markers["50%"]
    elif 100 <= phase_angle < 150:
        marker = moon_markers["30%"]
    elif 150 <= phase_angle < 175:
        marker = moon_markers["10%"]
    elif phase_angle >= 175:
        marker = moon_markers["0%"]
        color = "black"

    # XY PLANE
    # Plot the orbit
    axs[0].plot(orbit[:, 0], orbit[:, 1], color="red", zorder=1)

    # plot the black marker below
    axs[0].scatter(
        to_arcsec(x, d),
        to_arcsec(y, d),
        s=150,
        marker="o",
        color="black",
        edgecolor="black",
        zorder=2,
    )

    # Plot the planet with the appropriate marker
    axs[0].scatter(
        to_arcsec(x, d), to_arcsec(y, d), s=100, marker=marker, color=color, zorder=2
    )

    axs[0].scatter(0, 0, s=100, marker="*", color="black")
    axs[0].set_xlabel("X (arcsec)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
    axs[0].set_ylabel("Y (arcsec)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])

    axs[0].set_xlim(lims)
    axs[0].set_ylim([-0.5, 0.5])
    # YZ PLANE
    # Plot the orbit
    axs[1].plot(orbit[:, 2], orbit[:, 1], color="red", zorder=1)

    # plot the black marker below
    axs[1].scatter(
        to_arcsec(z, d),
        to_arcsec(y, d),
        s=150,
        marker="o",
        color="black",
        edgecolor="black",
        zorder=2,
    )

    # Plot the planet with the appropriate marker
    axs[1].scatter(
        to_arcsec(z, d), to_arcsec(y, d), s=100, marker=marker, color=color, zorder=2
    )

    axs[1].scatter(0, 0, s=100, marker="*", color="black")
    axs[1].set_xlabel("Z (arcsec)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
    axs[1].set_ylabel("Y (arcsec)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])

    axs[1].set_xlim(lims)
    axs[1].set_ylim([-0.5, 0.5])
    # Set shared title for the figure
    fig.suptitle(
        f"$\\mathbf{{{label}}}$: a={a} AU, $\\bar i$={i}°, f={t}°, $\\phi={phase_angle:.1f}°$",
        fontsize=HALF_WIDTH_SIZES["SUBTITLE_SIZE"],
    )
    plt.tight_layout()
    return fig


def plot_multiple_2d(
    matrix, tolerance, a_range, i_range, t_range, slice_var="true_anomaly", num_slices=6
):
    cold_colormap, warm_colormap, full_colormap = get_colormaps()

    plt.rcParams.update(
        {"font.size": HALF_WIDTH_SIZES["TICK_SIZE"]}
    )  # Set a base font size

    if slice_var == "true_anomaly":
        slice_values = np.linspace(t_range.min(), t_range.max(), num_slices)
    elif slice_var == "inclination":
        slice_values = np.linspace(i_range.min(), i_range.max(), num_slices)
    else:  # semi-major axis
        slice_values = np.linspace(a_range.min(), a_range.max(), num_slices)

    # Find global min and max for consistent color scaling
    vmin = 0.01
    vmax = np.max(matrix)

    # Create a logarithmic normalization
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, slice_value in enumerate(slice_values):
        ax = axs[i]
        if slice_var == "true_anomaly":
            slice_index = np.argmin(np.abs(t_range - slice_value))
            data = matrix[:, slice_index, :]
            x, y = a_range, i_range
            xlabel, ylabel = r"$a$ (AU)", r"$\bar{i}$ (degrees)"
            name = r"$f$"
            unit = r"$^\circ$"
        elif slice_var == "inclination":
            slice_index = np.argmin(np.abs(i_range - slice_value))
            data = matrix[slice_index, :, :]
            x, y = a_range, t_range
            xlabel, ylabel = r"$a$ (AU)", r"$f$ (degrees)"
            name = r"$\bar{i}$"
            unit = r"$^\circ$"

        else:  # semi-major axis
            slice_index = np.argmin(np.abs(a_range - slice_value))
            data = matrix[:, :, slice_index]
            y, x = i_range, t_range  # flipped around
            ylabel, xlabel = r"$\bar{i}$ (degrees)", r"$f$ (degrees)"
            name = r"$a$"
            unit = r" AU"

        im = ax.imshow(
            data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
            origin="lower",
            cmap=full_colormap,
            norm=norm,
        )
        # ax.contour(x, y, data, levels=[tolerance], colors='white', linewidths=1)

        ax.set_xlabel(xlabel, fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
        ax.set_ylabel(ylabel, fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
        ax.set_title(
            f"{name} = {slice_value:.2f}{unit}",
            fontsize=HALF_WIDTH_SIZES["SUBTITLE_SIZE"],
        )

    fig.tight_layout()

    # Add a colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Difference (arcsec)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])

    # Adjust the main subplot to make room for the colorbar
    plt.subplots_adjust(right=0.9)
    return fig


def plot_orbits_and_spectra(
    UpOmega,
    omega,
    d,
    orbit_metadata_file,
    spectrum_metadata_file,
    earth_orbit_file,
    earth_flux_file,
    lims=[[-2, 2], [-1, 1],[1e-13,1e-6]],
    inset=False,
    logscale=False,
    sigma=1,
):
    cold_colormap, warm_colormap, full_colormap = get_colormaps()
    # Create a colormap
    cmap = cold_colormap  # plt.get_cmap('GnBu_r')
    cmap2 = warm_colormap  # plt.get_cmap('Reds')
    count_internal = 0
    norm = Normalize(vmin=0, vmax=180)

    # Load metadata
    orbit_metadata = pd.read_csv(orbit_metadata_file)
    spectrum_metadata = pd.read_csv(spectrum_metadata_file)

    # Load Earth data
    earth_orbit = np.loadtxt(earth_orbit_file)
    earth_file = np.loadtxt(earth_flux_file)
    earth_flux = earth_file[:, -1] * u.W / u.m**2 / u.um
    wavelength = earth_file[:, 0]
    earth_flux = earth_flux.to(
        u.photon / (u.cm**2 * u.s * u.nm),
        equivalencies=u.spectral_density(wavelength * u.micron),
    )

    plt.rcParams.update({"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]})

    # ******** PLOT ORBITS *******
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 4, width_ratios=[10, 10, 0.7, 0.7], height_ratios=[0.7, 1.2], hspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :2])
    ca1 = fig.add_subplot(gs[:, 2])
    ca2 = fig.add_subplot(gs[:, 3])

    ax0.scatter(x=0, y=-0.1, color="yellow", s=40, zorder=104,lw=1, edgecolors='k')
    ax1.scatter(x=0, y=-0.1, color="yellow", s=40,zorder=104,lw=1, edgecolors='k')

    # Subplot 0: X vs Y
    ax0.plot(
        earth_orbit[:, 0],
        earth_orbit[:, 1],
        "k",
        label=r"Earth, a = 1 AU, i = 0$^\circ$",
        zorder=102,
    )
    ax0.set_xlabel("X (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax0.set_ylabel("Y (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax0.set_ylim(lims[1][0], lims[1][1])
    ax0.set_xlim(lims[0][0], lims[0][1])
    ax0.set_title("Orbital Plane (X vs Y)", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
    ax0.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

    # Subplot 1: Z vs Y
    ax1.plot(
        earth_orbit[:, 2],
        earth_orbit[:, 1],
        "k",
        label=r"Earth, a = 1 AU, i = 0$^\circ$",
        zorder=102,
    )
    ax1.set_xlabel("Z (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    # ax1.set_ylabel('Y (arcsec)', fontsize=LABEL_SIZE)
    ax1.set_ylim(lims[1][0], lims[1][1])
    ax1.set_xlim(lims[0][0], lims[0][1])
    ax1.set_title("Orbital Plane (Z vs Y)", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
    ax1.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

    if inset:
        # Create the inset axes
        axins = inset_axes(
            ax1, width="40%", height="40%", loc="lower center", borderpad=1.8
        )

        # Plot the zoomed-in view
        axins.plot(
            earth_orbit[:, 2],
            earth_orbit[:, 1],
            "k",
            label="Earth",
            zorder=102,
            linewidth=2,
        )

        # Set the limits for the inset plot
        axins.set_xlim(-0.12, 0.12)
        axins.set_ylim(-0.12, 0.12)

    # Subplot 2: Spectra
    ax2.plot(
        wavelength,
        earth_flux,
        "k",
        label=r"Earth, a = 1 AU, i = 0$^\circ$, phase angle = 90$^\circ$",
        lw=2,
        zorder=102,
    )
    lam_min_primary, lam_max_primary, dlambda = calculate_deltalambda_edges(0.5, 0.2, 5)
    earth_avg05 = find_average_fluxes(
        wavelength, earth_flux, [lam_min_primary, lam_max_primary]
    )
    earth_yerr = earth_avg05 / 7  # Error is 1/7 of the flux
    earth_lower = earth_avg05 - earth_yerr
    earth_upper = earth_avg05 + earth_yerr
    print("Earth average flux at 0.5 micron is " + str(earth_avg05))
    ax2.errorbar(
        0.5,
        earth_avg05,
        yerr=earth_yerr,
        xerr=dlambda/2,
        fmt="o",
        capsize=5,
        markersize=2,
        zorder=102,
        color='yellow',

    )

    ax2.set_xlabel("Wavelength (μm)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax2.set_ylabel(
        f"Flux [{earth_flux.unit.to_string('latex_inline')}]",
        fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
    )
    ax2.set_title("Spectra", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
    ax2.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])
    
    df = pd.DataFrame(
        columns=[
            "a (AU)",
            "i (deg)",
            "t (deg)",
            "Flux Overlap",
            "Earth Flux",
            "Neptune Flux",
        ]
    )
    # Plot orbits and spectra of the other planet
    for index, row in orbit_metadata.iterrows():
        orbit = np.loadtxt(row["file"])
        spectrum = np.loadtxt(
        spectrum_metadata.loc[
            spectrum_metadata["index"] == row["index"], "file"
        ].values[0]
        )
        flux = spectrum[:, 3] * u.W / u.m**2 / u.um
        wavelength = spectrum[:, 0]
        flux = flux.to(
            u.photon / (u.cm**2 * u.s * u.nm),
            equivalencies=u.spectral_density(wavelength * u.micron),
        )

        color = cmap(norm(row["i"]))
        color_2 = cmap2(norm(row["i"]))

        r = row["a"] * const.au
        inclination = np.radians(row["i"])
        true_anomaly = np.radians(row["t"])

        x, y, z = calculate_orbit_location(r, inclination, true_anomaly, UpOmega, omega)
        # ax0.scatter(to_arcsec(x, d), to_arcsec(y, d), color=color, s=5)
        # ax1.scatter(to_arcsec(z, d), to_arcsec(y, d), color=color, s=5)
        if inset:
            axins.scatter(to_arcsec(z, d), to_arcsec(y, d), color=color, s=10)

        if index == len(orbit_metadata) - 10:
            ax0.plot(
                orbit[:, 0],
                orbit[:, 1],
                color=color,
                lw=1,
                alpha=0.5,
                label="Planets that satisfy 1)",
            )
            ax1.plot(
                orbit[:, 2],
                orbit[:, 1],
                color=color,
                lw=1,
                alpha=0.5,
                label="Planets that satisfy 1)",
            )
            ax2.plot(
                wavelength,
                flux,
                ":",
                color=color,
                lw=1,
                alpha=0.5,
                label="Planets that satisfy 1)",
            )
        else:
            ax0.plot(
                orbit[:, 0],
                orbit[:, 1],
                color=color,
                lw=1,
                alpha=0.5,
            )
            ax1.plot(orbit[:, 2], orbit[:, 1], color=color, lw=1, alpha=0.5)
            ax2.plot(
                wavelength,
                flux,
                ":",
                color=color,
                lw=1,
                alpha=0.5,
            )
        if inset:
            axins.plot(orbit[:, 2], orbit[:, 1], color=color, alpha=0.5)

        # ### LOOK FOR SIMILAR FLUXES
        planet_avg05 = find_average_fluxes(
            wavelength, flux, [lam_min_primary, lam_max_primary]
        )
        # planet_yerr = planet_avg05 / 7  # Error is 1/7 of the flux

        # # Check if the error bars overlap

        # planet_lower = planet_avg05 - planet_yerr
        # planet_upper = planet_avg05 + planet_yerr

        # Overlap criterion: maximum value of minima - minimum value of maxima <= 0
        # if (max(earth_lower, planet_lower) - min(earth_upper, planet_upper)) <= 0:
            # flux_overlap = max(earth_lower, planet_lower) - min(
            #     earth_upper, planet_upper
            # )
        if earth_lower<=planet_avg05<=earth_upper:

            x, y, z = calculate_orbit_location(
                row["a"] * const.au,
                np.radians(row["i"]),
                np.radians(row["t"]),
                UpOmega,
                omega,
            )

            phase_angle = calculate_apparent_phase(x, y, z, d)

            # Append the values to the DataFrame
            new_row = pd.DataFrame(
                [
                    {
                        "a (AU)": row["a"],
                        "i (deg)": row["i"],
                        "t (deg)": row["t"],
                        "phi (deg)": phase_angle,
                        # "Flux Overlap": flux_overlap,
                        "Earth Flux": earth_avg05,
                        "Neptune Flux": planet_avg05,
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)
            if count_internal == 19:
                ax2.plot(
                    wavelength,
                    flux,
                    "-",
                    color=color_2,
                    lw=2,
                    alpha=0.5,
                    label="Neptunes that satisfy 1) and 2)",
                )
                ax0.plot(
                    orbit[:, 0],
                    orbit[:, 1],
                    "-",
                    color=color_2,
                    lw=2,
                    alpha=0.5,
                    zorder=101,
                    label="Neptunes that satisfy 1) and 2)",
                )
                ax1.plot(
                    orbit[:, 2],
                    orbit[:, 1],
                    "-",
                    color=color_2,
                    lw=2,
                    alpha=0.5,
                    zorder=101,
                    label="Neptunes that satisfy 1) and 2)",
                )

            else:
                ax2.plot(
                    wavelength,
                    flux,
                    "-",
                    color=color_2,
                    lw=2,
                    alpha=0.5,
                )
                ax0.plot(orbit[:, 0], orbit[:, 1], "-", color=color_2, lw=2, alpha=0.5,zorder=101)
                ax1.plot(orbit[:, 2], orbit[:, 1], "-", color=color_2, lw=2, alpha=0.5,zorder=101)
                if inset:
                    axins.plot(
                        orbit[:, 2], orbit[:, 1], "-", color=color_2, lw=2, alpha=0.5
                    )

            count_internal += 1

            # ax2.errorbar(0.5, neptune_avg05, yerr=neptune_yerr, color='red', fmt='o', capsize=5, markersize=5, zorder=101)

    # # Add observer arrow
    # arrow_start = (-1.4, -1.2)  # Starting point of the arrow
    # arrow_end = (-1.2, -1.2)    # Ending point of the arrow
    # ax1.annotate('Observer', xy=arrow_start, xytext=arrow_end,
    #             arrowprops=dict(arrowstyle='->', color='black', lw=2),
    #             ha='left', va='center', color='black')

    # Add legend
    # Create gradient-filled rectangles for legend
    gradient_width = 50
    gradient_height = 1

    # Create gradients
    gradient1 = np.linspace(0, 1, gradient_width).reshape(1, -1)
    gradient1 = np.vstack((gradient1, gradient1))
    colors1 = cmap(gradient1)

    gradient2 = np.linspace(0, 1, gradient_width).reshape(1, -1)
    gradient2 = np.vstack((gradient2, gradient2))
    colors2 = cmap2(gradient2)

    # Get the position of ax2
    ax2_pos = ax2.get_position()

    # Calculate the position for the legend axes (10% of ax2 width, in the top right corner)
    legend_width = ax2_pos.width * 0.52
    legend_height = ax2_pos.height * 0.2
    legend_left = ax2_pos.x1 - legend_width - 0.01  # Slight padding from the right edge
    legend_bottom = (
        ax2_pos.y1 - legend_height - 0.01
    )  # Slight padding from the top edge

    # Create a separate axes for the legend items
    legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
    legend_ax.axis("off")

    # Plot the gradient rectangles and the Earth line
    gradient_width = legend_width * 0.5  # Width of the gradient part
    text_start = gradient_width * 1.15  # Where the text should start

    # Plot the gradient rectangles and the Earth line
    legend_ax.imshow(colors1, aspect="auto", extent=[0, gradient_width, 0.7, 0.9])
    legend_ax.imshow(colors2, aspect="auto", extent=[0, gradient_width, 0.35, 0.55])
    legend_ax.plot([0, gradient_width], [0.1, 0.1], "k-", linewidth=2)
    legend_ax.plot(
        [0, 1], [0.15, 0.15], "white", linewidth=0
    )  # fake to keep the right dimensions

    fontsize = FULL_WIDTH_SIZES["LEGEND_SIZE"]  # Adjust this factor to change text size
    legend_ax.text(
        text_start,
        0.8,
        "Neptunes that satisfy 1)",
        ha="left",
        va="center",
        fontsize=fontsize,
    )
    legend_ax.text(
        text_start,
        0.45,
        "Neptunes that satisfy 1) and 2)",
        ha="left",
        va="center",
        fontsize=fontsize,
    )
    legend_ax.text(
        text_start,
        0.1,
        "Earth, a = 1 AU, i = 0°",
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    # Add a box around the legend
    legend_box = patches.Rectangle(
        (0, 0),
        1,
        1,
        fill=False,
        edgecolor="lightgray",
        linewidth=2,
        transform=legend_ax.transAxes,
    )
    legend_ax.add_patch(legend_box)

    # Add padding inside the box
    padding = 0.05
    legend_ax.set_xlim(-padding, 1 + padding)
    legend_ax.set_ylim(-padding, 1 + padding)

    # ax2.set_ylim(1e-26, 1e-16)
    ax2.set_xlim(0.2, 2)
    ax1.set_yticklabels([])

    if logscale:
        ax2.set_yscale("log")
    
    ax2.set_ylim(lims[2][0], lims[2][1])
    # Adjust the inset plot appearance
    if inset:
        axins.tick_params(axis="both", which="major", labelsize=10)

        # # After the loop, add the colorbar
        # cax = inset_axes(
        #     ax1,
        #     width="5%",  # width of colorbar
        #     height="100%",  # height of colorbar
        #     loc="center left",
        #     bbox_to_anchor=(1.05, 0.0, 1, 1),
        #     bbox_transform=axins.transAxes,
        #     borderpad=0,
        # )
        # norm = Normalize(vmin=min(orbit_metadata["i"]), vmax=max(orbit_metadata["i"]))

        # cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        # cb.set_label(
        #     "Inclination (deg)",
        #     rotation=270,
        #     labelpad=8,
        #     fontsize=FULL_WIDTH_SIZES["LEGEND_SIZE"],
        # )
        # # Set the ticks to 0 and 180
        # cb.set_ticks([0, 180])
        # cb.set_ticklabels(["0", "180"], fontsize=10)
    # After the loop, add the colorbar

    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ca1)
    cb.set_ticklabels(
        [],
    )
    # After the loop, add the colorbar
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap2), cax=ca2)
    cb.set_label(
        "Inclination (degrees)",
        rotation=270,
        labelpad=15,
        fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
    )

    return df, fig


def plot_spectra(
    UpOmega,
    omega,
    d,
    orbit_metadata_file,
    spectrum_metadata_file,
    earth_orbit_file,
    earth_flux_file,
    lims=[[-2, 2], [-1, 1]],
    colormap="warm",
):
    cold_colormap, warm_colormap, full_colormap = get_colormaps()
    # Create a colormap
    cmap = plt.get_cmap("Greys_r")
    if colormap == "warm":
        cmap2 = warm_colormap  # plt.get_cmap('Reds')
    else:
        cmap2 = cold_colormap

    count_internal = 0

    # Load metadata
    orbit_metadata = pd.read_csv(orbit_metadata_file)

    spectrum_metadata = pd.read_csv(spectrum_metadata_file)
    norm = Normalize(vmin=min(orbit_metadata["a"]), vmax=max(orbit_metadata["a"]))

    # Load Earth data
    earth_file = np.loadtxt(earth_flux_file)
    earth_flux = earth_file[:, -1] * u.W / u.m**2 / u.um
    wavelength = earth_file[:, 0]
    earth_flux = earth_flux.to(
        u.photon / (u.cm**2 * u.s * u.nm),
        equivalencies=u.spectral_density(wavelength * u.micron),
    )

    plt.rcParams.update({"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]})

    # ******** PLOT ORBITS *******
    fig = plt.figure(figsize=(12, 6))

    # Subplot 2: Spectra
    plt.semilogy(
        wavelength,
        earth_flux,
        "k",
        label=r"Earth, a = 1 AU, i = 0$^\circ$, $\phi$ = 90$^\circ$",
        lw=2,
        zorder=100,
    )
    lam_min_primary, lam_max_primary, _ = calculate_deltalambda_edges(0.5, 0.2, 5)
    earth_avg05 = find_average_fluxes(
        wavelength, earth_flux, [lam_min_primary, lam_max_primary]
    )
    earth_yerr = earth_avg05 / 7  # Error is 1/7 of the flux

    print("Earth average flux at 0.5 micron is " + str(earth_avg05))
    plt.errorbar(
        0.5,
        earth_avg05,
        yerr=earth_yerr,
        color="k",
        fmt="o",
        capsize=5,
        markersize=5,
        zorder=101,
    )

    df = pd.DataFrame(
        columns=[
            "a (AU)",
            "i (deg)",
            "t (deg)",
            "Flux Overlap",
            "Earth Flux",
            "Neptune Flux",
        ]
    )
    # Plot orbits and spectra of the other planet
    for index, row in orbit_metadata.iterrows():

        orbit = np.loadtxt(row["file"])
        spectrum = np.loadtxt(
            spectrum_metadata.loc[
                spectrum_metadata["index"] == row["index"], "file"
            ].values[0]
        )
        flux = spectrum[:, -1] * u.W / u.m**2 / u.um
        wavelength = spectrum[:, 0]
        flux = flux.to(
            u.photon / (u.cm**2 * u.s * u.nm),
            equivalencies=u.spectral_density(wavelength * u.micron),
        )

        color = cmap(norm(row["a"]))
        color_2 = cmap2(norm(row["a"]))

        r = row["a"] * const.au
        inclination = np.radians(row["i"])
        true_anomaly = np.radians(row["t"])

        x, y, z = calculate_orbit_location(r, inclination, true_anomaly, UpOmega, omega)

        if index == int(len(orbit_metadata) / 16):
            plt.plot(
                wavelength,
                flux,
                "--",
                color=color,
                lw=1,
                alpha=0.2,
                label="Planets that satisfy 1)",
            )
        else:
            plt.plot(
                wavelength,
                flux,
                "--",
                color=color,
                lw=1,
                alpha=0.2,
            )

        ### LOOK FOR SIMILAR FLUXES
        planet_avg05 = find_average_fluxes(
            wavelength, flux, [lam_min_primary, lam_max_primary]
        )
        planet_yerr = planet_avg05 / 7  # Error is 1/7 of the flux

        # Check if the error bars overlap
        earth_lower = earth_avg05 - earth_yerr
        earth_upper = earth_avg05 + earth_yerr
        planet_lower = planet_avg05 - planet_yerr
        planet_upper = planet_avg05 + planet_yerr

        # Overlap criterion: maximum value of minima - minimum value of maxima <= 0
        if (max(earth_lower, planet_lower) - min(earth_upper, planet_upper)) <= 0:
            flux_overlap = max(earth_lower, planet_lower) - min(
                earth_upper, planet_upper
            )

            phase_angle = calculate_apparent_phase(x, y, z, d)

            # Append the values to the DataFrame
            new_row = pd.DataFrame(
                [
                    {
                        "a (AU)": row["a"],
                        "i (deg)": row["i"],
                        "t (deg)": row["t"],
                        "phi (deg)": phase_angle,
                        "Flux Overlap": flux_overlap,
                        "Earth Flux": earth_avg05,
                        "Neptune Flux": planet_avg05,
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)
            if count_internal == 0:
                plt.plot(
                    spectrum[:, 0],
                    spectrum[:, -1],
                    "-",
                    color=color_2,
                    lw=2,
                    alpha=0.5,
                    label="Neptunes that satisfy 1) and 2)",
                )

            else:
                plt.plot(
                    spectrum[:, 0],
                    spectrum[:, -1],
                    "-",
                    color=color_2,
                    lw=2,
                    alpha=0.5,
                )

            count_internal += 1

    plt.xlabel("Wavelength (μm)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    plt.ylabel(
        f"Flux [{spectrum.unit.to_string('latex_inline')}]",
        fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
    )
    plt.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

    plt.legend()
    plt.ylim(1e-26, 1e-16)
    plt.xlim(0.2, 2)

    # plt.savefig("imgs/orbits_and_spectra_Wm2um.pdf")
    return df, fig


def plot_orbits(
    UpOmega,
    omega,
    d,
    orbit_metadata_file,
    earth_orbit_file,
    earth_flux_file,
    lims=[[-2, 2], [-1, 1]],
):
    cold_colormap, warm_colormap, full_colormap = get_colormaps()
    # Create a colormap
    cmap = full_colormap.reversed()  # Invert the colormap
    count_internal = 0

    # Load metadata
    orbit_metadata = pd.read_csv(orbit_metadata_file)
    norm = LogNorm(vmin=min(orbit_metadata["a"]), vmax=max(orbit_metadata["a"]))
    # Load Earth data
    earth_orbit = np.loadtxt(earth_orbit_file)

    plt.rcParams.update({"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]})

    # ******** PLOT ORBITS *******
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[20, 1], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])
    ax0.scatter(x=0, y=-0.1, color="k", zorder=101)
    ax1.scatter(x=0, y=-0.1, color="k", zorder=101)

    # Subplot 0: X vs Y
    ax0.plot(
        earth_orbit[:, 0],
        earth_orbit[:, 1],
        "k",
        label=r"Earth, a = 1 AU, i = 0$^\circ$",
        zorder=101,
    )
    ax0.set_xlabel("X (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax0.set_ylabel("Y (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax0.set_ylim(lims[1][0], lims[1][1])
    ax0.set_xlim(lims[0][0], lims[0][1])
    ax0.set_title("Orbital Plane (X vs Y)", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
    ax0.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

    # Subplot 1: Z vs Y
    ax1.plot(
        earth_orbit[:, 2],
        earth_orbit[:, 1],
        "k",
        label=r"Earth, a = 1 AU, i = 0$^\circ$",
        zorder=101,
    )
    ax1.set_xlabel("Z (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax1.set_ylabel("Y (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
    ax1.set_ylim(lims[1][0], lims[1][1])
    ax1.set_xlim(lims[0][0], lims[0][1])
    ax1.set_title("Orbital Plane (Z vs Y)", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
    ax1.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

    # Plot orbits and spectra of the other planet
    for index, row in orbit_metadata.iterrows():
        orbit = np.loadtxt(row["file"])

        color = cmap(norm(row["a"]))

        r = row["a"] * const.au
        inclination = np.radians(row["i"])
        true_anomaly = np.radians(row["t"])

        x, y, z = calculate_orbit_location(r, inclination, true_anomaly, UpOmega, omega)
        ax0.scatter(to_arcsec(x, d), to_arcsec(y, d), color=color, s=5)
        ax1.scatter(to_arcsec(z, d), to_arcsec(y, d), color=color, s=5)
        if index == len(orbit_metadata) - 10:
            ax0.plot(
                orbit[:, 0],
                orbit[:, 1],
                color=color,
                lw=1,
                alpha=0.5,
                label="Planets that satisfy 1)",
            )
            ax1.plot(
                orbit[:, 2],
                orbit[:, 1],
                color=color,
                lw=1,
                alpha=0.5,
                label="Planets that satisfy 1)",
            )

        else:
            ax0.plot(
                orbit[:, 0],
                orbit[:, 1],
                color=color,
                lw=1,
                alpha=0.5,
            )
            ax1.plot(orbit[:, 2], orbit[:, 1], color=color, lw=1, alpha=0.5)

    # After the loop, add the colorbar
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(
        "Semi-major Axis (AU)",
        rotation=270,
        labelpad=15,
        fontsize=FULL_WIDTH_SIZES["LEGEND_SIZE"],
    )
    # Set the ticks to 0 and 180
    cb.set_ticks([1, 3, 10, max(orbit_metadata["a"])])
    cb.set_ticklabels(
        ["1", "3", "10", str(max(orbit_metadata["a"]))],
        fontsize=FULL_WIDTH_SIZES["TICK_SIZE"],
    )
    plt.tight_layout()
    return fig


# def plot_best_neptunes(df, UpOmega, omega, d):
#     plt.rcParams.update(
#         {"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]}
#     )  # Set a base font size
#     cold_colormap, warm_colormap, full_colormap = get_colormaps()

#     # ******** PLOT OF BEST ONES *******
#     fig = plt.figure(figsize=(12, 12))
#     gs = GridSpec(2, 2, height_ratios=[1, 1.2])

#     ax0 = fig.add_subplot(gs[0, 0])
#     ax1 = fig.add_subplot(gs[0, 1])
#     ax2 = fig.add_subplot(gs[1, :])

#     earth_orbit = calculate_full_orbit(
#         r=const.au,
#         inclination=np.radians(0),
#         UpOmega=np.radians(0),
#         omega=np.radians(90),
#         d=(10 * u.pc).to(u.m),
#         save=True,
#         filename="Earth_at_quadrature/orbit.txt",
#     )
#     x, y, z = calculate_orbit_location(
#         r=const.au,
#         inclination=np.radians(0),
#         true_anomaly=np.radians(0),
#         UpOmega=np.radians(0),
#         omega=np.radians(90),
#     )
#     ax0.scatter(x=0, y=-0.1, color="k", zorder=101)
#     ax1.scatter(x=0, y=-0.1, color="k", zorder=101)

#     # Subplot 0: X vs Y
#     ax0.plot(
#         earth_orbit[:, 0],
#         earth_orbit[:, 1],
#         "k",
#         label=r"Earth, a = 1 AU, i = 0$^\circ$",
#         zorder=101,
#     )
#     ax0.set_xlabel("X (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
#     ax0.set_ylabel("Y (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
#     ax0.set_ylim(-1, 1)
#     ax0.set_xlim(-1.5, 1.5)
#     ax0.set_title("Orbital Plane (X vs Y)", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
#     ax0.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

#     # Subplot 1: Z vs Y
#     ax1.plot(
#         earth_orbit[:, 2],
#         earth_orbit[:, 1],
#         "k",
#         label=r"Earth, a = 1 AU, i = 0$^\circ$",
#         zorder=101,
#     )
#     ax1.set_xlabel("Z (arcsec)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
#     # ax1.set_ylabel('Y (arcsec)', fontsize=LABEL_SIZE)
#     ax1.set_ylim(-1, 1)
#     ax1.set_xlim(-1.5, 1.5)
#     ax1.set_title("Orbital Plane (Z vs Y)", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
#     ax1.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

#     # Subplot 2: Spectra
#     earth_flux = np.loadtxt(
#         "Earth_at_quadrature/psg_rad_Wm2um_subsampling5.txt"
#     )  # calculate_flux('Earth_at_quadrature/Earth_summer_avg_Wm2um.txt', 'Earth_at_quadrature/psg_rad_Wm2um.txt')
#     ax2.semilogy(
#         earth_flux[:, 0],
#         earth_flux[:, -1],
#         "k",
#         label=r"Earth, a = 1 AU, i = 0$^\circ$, phase angle = 90$^\circ$",
#         lw=1,
#         zorder=100,
#     )
#     lam_min_primary, lam_max_primary, _ = calculate_deltalambda_edges(0.5, 0.2, 5)
#     earth_avg05 = find_average_fluxes(
#         earth_flux[:, 0], earth_flux[:, -1], [lam_min_primary, lam_max_primary]
#     )
#     earth_yerr = earth_avg05 / 7  # Error is 1/7 of the flux

#     ax2.errorbar(
#         0.5,
#         earth_avg05,
#         yerr=earth_yerr,
#         color="k",
#         fmt="o",
#         capsize=5,
#         markersize=5,
#         zorder=101,
#     )
#     ax2.set_xlabel("Wavelength (μm)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
#     ax2.set_ylabel("Fp (W/m2/μm)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
#     ax2.set_title("Spectra", fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"])
#     ax2.tick_params(axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"])

#     selected_i = df["i (deg)"].values
#     selected_t = df["t (deg)"].values
#     selected_a = df["a (AU)"].values

#     # Create a colormap
#     cmap = warm_colormap  #  plt.get_cmap('autumn_r')
#     norm = Normalize(vmin=40, vmax=100)

#     for index in range(len(selected_i)):
#         color = cmap(norm(selected_i[index]))
#         r = selected_a[index] * const.au
#         inclination = np.radians(selected_i[index])
#         true_anomaly = np.radians(selected_t[index])

#         x, y, z = calculate_orbit_location(r, inclination, true_anomaly, UpOmega, omega)
#         ax0.scatter(to_arcsec(x, d), to_arcsec(y, d), color=color, s=5)
#         ax1.scatter(to_arcsec(z, d), to_arcsec(y, d), color=color, s=5)

#         orbit = calculate_full_orbit(r, inclination, UpOmega, omega, d)
#         if index == 19:
#             ax0.plot(
#                 orbit[:, 0],
#                 orbit[:, 1],
#                 color=color,
#                 lw=1,
#                 label="Neptunes that satisfy 1) and 2)",
#             )
#             ax1.plot(
#                 orbit[:, 2],
#                 orbit[:, 1],
#                 color=color,
#                 lw=1,
#                 label="Neptunes that satisfy 1) and 2)",
#             )
#         else:
#             ax0.plot(
#                 orbit[:, 0],
#                 orbit[:, 1],
#                 color=color,
#                 lw=1,
#             )
#             ax1.plot(
#                 orbit[:, 2],
#                 orbit[:, 1],
#                 color=color,
#                 lw=1,
#             )

#         r = selected_a[index]
#         i = selected_i[index]
#         t = selected_t[index]

#         neptune_flux = calculate_and_store_spectrum(
#             "Neptune/psg_NEWTEMPLATE_Wm2um.txt",
#             "Fixed_UpOmega/spectra/Wm2um",
#             r,
#             i,
#             t,
#             recalculate=False,
#         )
#         if index == 19:
#             ax2.plot(
#                 neptune_flux[:, 0],
#                 neptune_flux[:, -1],
#                 "-.",
#                 color=color,
#                 lw=1,
#                 label="Neptunes that satisfy 1) and 2)",
#             )
#         else:
#             ax2.plot(neptune_flux[:, 0], neptune_flux[:, -1], "-.", color=color, lw=1)

#     # # Add observer arrow
#     # arrow_start = (-1.4, -1.2)  # Starting point of the arrow
#     # arrow_end = (-1.2, -1.2)    # Ending point of the arrow
#     # ax1.annotate('Observer', xy=arrow_start, xytext=arrow_end,
#     #             arrowprops=dict(arrowstyle='->', color='black', lw=2),
#     #             ha='left', va='center', color='black')

#     # Create an invisible axins
#     axins = inset_axes(ax1, width="40%", height="40%", loc="lower center", borderpad=1)
#     axins.set_axis_off()

#     # After the loop, add the colorbar
#     cax = inset_axes(
#         ax1,
#         width="5%",  # width of colorbar
#         height="100%",  # height of colorbar
#         loc="center left",
#         bbox_to_anchor=(1.30, 0.0, 1, 1),
#         bbox_transform=axins.transAxes,
#         borderpad=0,
#     )

#     cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
#     cb.set_label(
#         "Inclination (deg)",
#         rotation=270,
#         labelpad=8,
#         fontsize=FULL_WIDTH_SIZES["LEGEND_SIZE"],
#     )
#     # Set the ticks to min and max of selected_i
#     cb.set_ticks([40, 100])
#     cb.set_ticklabels([f"40", f"100"], fontsize=10)

#     def gradient_rectangle(cmap, label):
#         gradient = np.linspace(0, 1, 256).reshape(1, -1)
#         gradient = np.vstack((gradient, gradient))
#         return Rectangle((0, 0), 1, 1), label

#     # Create legend entries
#     earth_line = Line2D([0], [0], color="k", label=r"Earth, a = 1 AU, i = 0$^\circ$")
#     neptune_satisfy_1_2, label_2 = gradient_rectangle(
#         cmap, "Neptunes that satisfy 1) and 2)"
#     )

#     # Create proxy artists for the legend
#     neptune_satisfy_1_2.set_facecolor(cmap(0.5))

#     # Add legends to the subplots
#     legend_elements = [earth_line, neptune_satisfy_1_2]
#     legend_labels = [earth_line.get_label(), label_2]

#     ax2.legend(
#         legend_elements,
#         legend_labels,
#         loc="upper right",
#         fontsize=FULL_WIDTH_SIZES["LEGEND_SIZE"],
#     )
#     ax0.legend(
#         legend_elements,
#         legend_labels,
#         loc="lower right",
#         fontsize=FULL_WIDTH_SIZES["LEGEND_SIZE"],
#     )

#     # ax2.legend(loc='upper right', fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
#     # #ax1.legend(loc='upper right')
#     # ax0.legend(loc='lower right', fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
#     ax2.set_ylim(1e-26, 1e-16)
#     ax2.set_xlim(0.2, 2)

#     return fig
#     # plt.savefig("imgs/best_neptunes_Wm2um.pdf")


# def plot_spectrum_comparison(
#     observed_earth_bandwidths, observed_far_neptune_bandwidths
# ):

#     plt.rcParams.update(
#         {"font.size": HALF_WIDTH_SIZES["TICK_SIZE"]}
#     )  # Set base font size
#     cold_colormap, warm_colormap, full_colormap = get_colormaps()
#     fig = plt.figure(figsize=(12, 16))
#     gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])

#     bandwidths = [0.05, 0.1, 0.2]
#     colors = {
#         "Earth": "black",
#         "Far Neptune": full_colormap(0.2),
#         "Close Neptune": full_colormap(0.8),
#     }
#     markers = {"Earth": "o", "Far Neptune": "s", "Close Neptune": "d"}
#     lss = {0.05: ":", 0.1: "--", 0.2: "-"}

#     for i, band in enumerate(bandwidths):
#         ax = fig.add_subplot(gs[i, 0])
#         earth_df = observed_earth_bandwidths[observed_earth_bandwidths["frame"] == band]
#         far_neptune_df = observed_far_neptune_bandwidths[
#             observed_far_neptune_bandwidths["frame"] == band
#         ]

#         for planet, df in [("Earth", earth_df), ("Far Neptune", far_neptune_df)]:
#             ax.errorbar(
#                 np.array(df["wavelength"]),
#                 np.array(df["flux"]),
#                 yerr=np.array(df["flux"] / df["snr"]),
#                 color=colors[planet],
#                 fmt=markers[planet],
#                 label=planet,
#                 alpha=1,
#                 capsize=3,
#                 markersize=6,
#                 elinewidth=1.5,
#             )

#         ax.set_ylabel("Flux", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
#         ax.set_title(
#             f"Bandwidth: {int(band*100)}%", fontsize=HALF_WIDTH_SIZES["SUBTITLE_SIZE"]
#         )
#         ax.legend(
#             fontsize=HALF_WIDTH_SIZES["LEGEND_SIZE"],
#             loc="lower left",
#             frameon=True,
#             fancybox=False,
#             edgecolor="lightgray",
#             facecolor="none",
#             borderpad=0.5,
#         )
#         ax.grid(True, which="both", linestyle="--", alpha=0.3)
#         # ax.set_yscale('log')
#         ax.tick_params(
#             axis="both", which="major", labelsize=HALF_WIDTH_SIZES["TICK_SIZE"]
#         )
#         ax.set_ylim(0, 2e-10)

#         # Remove x-axis labels for all subplots except the bottom one
#         if i < 2:
#             ax.set_xlabel("")
#             plt.setp(ax.get_xticklabels(), visible=False)
#         else:
#             ax.set_xlabel("Wavelength (μm)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])

#     # ax_diff = fig.add_subplot(gs[3, 0])
#     # for band in bandwidths:
#     #     earth_df = observed_earth_bandwidths[observed_earth_bandwidths['frame'] == band]
#     #     far_neptune_df = observed_far_neptune_bandwidths[observed_far_neptune_bandwidths['frame'] == band]
#     #     close_neptune_df = observed_close_neptune_bandwidths[observed_close_neptune_bandwidths['frame'] == band]

#     #     mean_squared_error =np.sqrt(np.array(np.array([e.value for e in earth_df.error]))**2+np.array(np.array([n.value for n in neptune_df.error]))**2)
#     #     ax_diff.plot(earth_df['wavelength'],
#     #                  abs(earth_df['flux'] - neptune_df['flux']) / mean_squared_error,
#     #                  label=f'{band}', ls=lss[band], linewidth=2)

#     # ax_diff.set_ylabel(r'$\frac{|F_{Earth}-F_{Neptune}|}{MSE_{Earth+Neptune}}$', fontsize=LABEL_SIZE)
#     # ax_diff.set_xlabel("Wavelength (μm)", fontsize=LABEL_SIZE)
#     # ax_diff.legend(fontsize=LEGEND_SIZE, loc='upper right')
#     # ax_diff.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
#     # fig.suptitle('Comparison of Earth and Neptune Spectra', fontsize=TITLE_SIZE, y=0.98)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9, hspace=0.3)

#     return fig


def plot_spectrum_comparison_shadedareas(
    observed_earth_bandwidths, observed_planet_bandwidths, label
):
    cold_colormap, warm_colormap, full_colormap = get_colormaps()
    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])
    bandwidths = [0.05, 0.1, 0.2]
    colors = {
        "Earth": "black",
        "Cold Neptune": full_colormap(0.2),
        "Warm Neptune": full_colormap(0.8),
    }
    markers = {"Earth": "o", "Cold Neptune": "s", "Warm Neptune": "d"}

    for i, band in enumerate(bandwidths):
        ax = fig.add_subplot(gs[i, 0])
        earth_df = observed_earth_bandwidths[observed_earth_bandwidths["frame"] == band]
        planet_df = observed_planet_bandwidths[
            observed_planet_bandwidths["frame"] == band
        ]

        for planet, df in [("Earth", earth_df), (label, planet_df)]:
            wavelength = np.array(df["wavelength"].astype(float))
            flux = np.array(df["flux"].astype(float))
            error = np.array(df["flux"].astype(float) / df["snr"].astype(float))
            ax.errorbar(
                wavelength,
                flux,
                color=colors[planet],
                marker=markers[planet],
                yerr=error,
                alpha=0.3,
                markersize=6,
                linestyle="",
            )
            ax.plot(
                wavelength,
                flux,
                color=colors[planet],
                marker=markers[planet],
                label=planet,
                alpha=1,
                markersize=6,
                linestyle="",
            )
            if planet == "Earth":
                alpha = 0.1
            else:
                alpha = 0.3
            ax.fill_between(
                wavelength,
                flux - error,
                flux + error,
                color=colors[planet],
                alpha=alpha,
            )

        ax.set_ylabel("Flux", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
        ax.set_title(
            f"Bandwidth: {int(band*100)}%", fontsize=HALF_WIDTH_SIZES["SUBTITLE_SIZE"]
        )
        ax.legend(
            fontsize=HALF_WIDTH_SIZES["LEGEND_SIZE"],
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="lightgray",
            facecolor="none",
            borderpad=0.5,
        )
        # ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.tick_params(
            axis="both", which="major", labelsize=HALF_WIDTH_SIZES["TICK_SIZE"]
        )
        # ax.set_ylim(0, 2e-10)

        if i < 2:
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Wavelength (μm)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)
    return fig


def count_rates_comparison(Crp_list, title):
    cold_colormap, warm_colormap, full_colormap = get_colormaps()

    colors = [
        full_colormap(0.2),
        full_colormap(0.6),
        full_colormap(0.8),
    ]  # Colors for each bandwidth

    Crp = pd.DataFrame(Crp_list)
    Crp = Crp.T
    Crp.columns = ["5%", "10%", "20%"]

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

    # Plot count rates
    for index, column in enumerate(Crp.columns):
        ax1.plot(
            Crp.index,
            Crp[column],
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"{column}",
            color=colors[index],
        )

    ax1.set_ylabel("Planet Count Rates", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
    # ax1.set_title('Planet Count Rates for Different Bandwidths', fontsize=SUBTITLE_SIZE)
    ax1.legend(
        fontsize=HALF_WIDTH_SIZES["LEGEND_SIZE"],
        frameon=True,
        fancybox=False,
        edgecolor="lightgray",
        facecolor="none",
        borderpad=0.5,
    )
    ax1.set_yscale("log")
    ax1.tick_params(axis="both", which="major", labelsize=HALF_WIDTH_SIZES["TICK_SIZE"])

    # # Plot ratios
    # ax2.plot(Crp.index, Crp['10%']/Crp['5%'], marker='o', linewidth=2, markersize=8, label='10%/5%')
    # ax2.plot(Crp.index, Crp['20%']/Crp['5%'], marker='o', linewidth=2, markersize=8, label='20%/5%')

    ax1.set_xlabel("Wavelength (μm)", fontsize=HALF_WIDTH_SIZES["LABEL_SIZE"])
    # ax2.set_ylabel('Count Rate Ratio', fontsize=LABEL_SIZE)
    # ax2.set_title('Ratios of Planet Count Rates Relative to 5% Bandwidth', fontsize=SUBTITLE_SIZE)
    # ax2.legend(fontsize=LEGEND_SIZE)
    # ax2.set_ylim(0, 10)
    # ax2.tick_params(axis='both', which='major', labelsize=LEGEND_SIZE)

    # # Add grid to both subplots
    # ax1.grid(True, linestyle='--', alpha=0.7)
    # ax2.grid(True, linestyle='--', alpha=0.7)
    fig.suptitle(title, fontsize=HALF_WIDTH_SIZES["TITLE_SIZE"])
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)
    return fig


def plot_combined_bandwidth_results(
    flux_file,
    bandwidths,
    observed_planet_dict,
    count_rates_dict,
    title,
    use_planet_flux=True,
):
    flux = np.loadtxt(flux_file)
    wavelength = flux[:, 0]
    stellar_flux = flux[:, 2] * u.W / u.m**2 / u.um
    stellar_flux = stellar_flux.to(
        u.photon / (u.cm**2 * u.s * u.nm),
        equivalencies=u.spectral_density(flux[:, 0] * u.micron),
    )
    planet_contrast = flux[:, -1] / flux[:, 2]
    planet_flux = flux[:, -1] * u.W / u.m**2 / u.um  # restore value
    planet_flux = planet_flux.to(
        u.photon / (u.cm**2 * u.s * u.nm),
        equivalencies=u.spectral_density(flux[:, 0] * u.micron),
    )

    cold_colormap, warm_colormap, full_colormap = get_colormaps()

    plt.rcParams.update(
        {"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]}
    )  # Set base font size

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    colors = [
        full_colormap(0.2),
        full_colormap(0.6),
        full_colormap(0.8),
    ]  # Colors for each bandwidth

    for i, bandwidth in enumerate(bandwidths):
        observed_planet = observed_planet_dict[bandwidth]
        count_rates = count_rates_dict[bandwidth]
        color = colors[i]

        # Planet spectrum
        if use_planet_flux:
            axs[0, i].plot(wavelength, planet_flux, c="gray", alpha=0.7, lw=1)
            y_data = observed_planet["flux"]
            y_label = f"Flux [{planet_flux.unit.to_string('latex_inline')}]"
        else:
            axs[0, i].plot(wavelength, planet_contrast, c="gray", alpha=0.7, lw=1)
            y_data = observed_planet["Fp/Fs"]
            y_label = "Fp/Fs"

        axs[0, i].scatter(
            observed_planet.wavelength,
            y_data,
            color=color,
            alpha=0.7,
            s=5,
        )
        axs[0, i].errorbar(
            observed_planet.wavelength,
            y_data,
            xerr=bandwidth / 2,
            color=color,
            alpha=0.7,
            fmt="o",
            capsize=5,
            markersize=5,
            label="Photometric bin",
        )
        axs[0, i].set_title(
            f"Bandwidth: {int(bandwidth*100)}%",
            fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"],
        )
        axs[0, i].legend(
            frameon=True,
            fancybox=False,
            edgecolor="lightgray",
            facecolor="none",
            borderpad=0.5,
        )
        if i == 0:
            axs[0, i].set_ylabel(y_label, fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
        axs[0, i].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )

        # axs[1, i].plot(
        #     np.array(observed_planet.wavelength),
        #     np.array(observed_planet.snr),
        #     color=color,
        #     alpha=0.3,
        #     lw=2,
        # )
        axs[1, i].step(
            np.array(observed_planet.wavelength),
            np.array(observed_planet.snr),
            color=color,
            where="mid",
        )
        axs[1, i].set_xlabel("Wavelength (μm)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
        if i == 0:
            axs[1, i].set_ylabel("SNR", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
        axs[1, i].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )

    # Set the same y-axis limits for all SNR plots
    snr_min = min(min(observed_planet_dict[bw].snr) for bw in bandwidths)
    snr_max = max(max(observed_planet_dict[bw].snr) for bw in bandwidths)
    for i in range(3):
        axs[1, i].set_ylim(snr_min, 1.2 * snr_max)
    fig.suptitle(title, fontsize=FULL_WIDTH_SIZES["TITLE_SIZE"])
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.2)
    return fig


def plot_snr_plot(
    bandwidths,
    observed_planet_dict,
    observed_earth_dict,
    title,
    label,
    use_planet_flux=True,
):
    planet_flux_unit = 1 * u.photon / (u.cm**2 * u.s * u.nm)
    output = {}
    cold_colormap, warm_colormap, full_colormap = get_colormaps()
    colors = {
        "Earth": "black",
        "Cold Neptune": full_colormap(0.2),
        "Warm Neptune": full_colormap(0.8),
    }
    markers = {"Earth": "o", "Cold Neptune": "s", "Warm Neptune": "d"}

    plt.rcParams.update(
        {"font.size": FULL_WIDTH_SIZES["TICK_SIZE"]}
    )  # Set base font size

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    for i, bandwidth in enumerate(bandwidths):
        output[bandwidth] = {}
        observed_planet = observed_planet_dict[bandwidth]
        observed_earth = observed_earth_dict[bandwidth]
        wavelength = np.array(observed_earth["wavelength"].astype(float))
        earth_flux = np.array(observed_earth["flux"].astype(float))
        earth_error = np.array(
            observed_earth["flux"].astype(float) / observed_earth["snr"].astype(float)
        )
        planet_flux = np.array(observed_planet["flux"].astype(float))
        planet_error = np.array(
            observed_planet["flux"].astype(float) / observed_planet["snr"].astype(float)
        )

        # Top row: Planet spectrum
        if use_planet_flux:
            y_data_planet = planet_flux
            y_data_earth = earth_flux
            y_label = f"Flux [{planet_flux_unit.unit.to_string('latex_inline')}]"
        else:
            y_data_planet = np.array(observed_planet["Fp/Fs"].astype(float))
            y_data_earth = np.array(observed_earth["Fp/Fs"].astype(float))
            y_label = "Fp/Fs"

        axs[0, i].errorbar(
            wavelength,
            y_data_planet,
            yerr=planet_error,
            xerr=bandwidth / 2,
            color=colors[label],
            marker=markers[label],
            alpha=0.7,
            markersize=6,
            label=label,
        )

        axs[0, i].fill_between(
            wavelength,
            y_data_planet - planet_error,
            y_data_planet + planet_error,
            color=colors[label],
            alpha=0.3,
        )

        axs[0, i].errorbar(
            wavelength,
            y_data_earth,
            yerr=earth_error,
            xerr=bandwidth / 2,
            color=colors["Earth"],
            marker=markers["Earth"],
            alpha=0.7,
            markersize=6,
            label="Earth",
        )
        axs[0, i].fill_between(
            wavelength,
            y_data_earth - earth_error,
            y_data_earth + earth_error,
            color=colors[label],
            alpha=0.1,
        )

        axs[0, i].set_title(
            f"Bandwidth: {int(bandwidth*100)}%",
            fontsize=FULL_WIDTH_SIZES["SUBTITLE_SIZE"],
        )

        if i == 0:
            axs[0, i].set_ylabel(y_label, fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
        axs[0, i].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )
        if i == 2:
            axs[0, i].legend()
        # Bottom row: Flux difference / combined error
        flux_diff = np.abs(y_data_earth - y_data_planet)

        combined_error = np.sqrt(earth_error**2 + planet_error**2)
        difference = flux_diff / combined_error
        snr = np.sqrt(np.sum(difference**2))

        lambdas_maxsignificance, max_significances, bandpasses = max_snr_finder(
            wavelength, difference, bandwidth
        )
        output[bandwidth]["lambdas"] = lambdas_maxsignificance
        output[bandwidth]["max_s"] = max_significances
        output[bandwidth]["bandpasses"] = bandpasses
        # # Add SNR text to the second subplot
        # axs[1, i].text(
        #     0.95,
        #     0.95,
        #     f"Band-integrated SNR: {snr:.2f}",
        #     transform=axs[1, i].transAxes,
        #     horizontalalignment="right",
        #     verticalalignment="top",
        #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        # )
        # # Find the indices of the top 10 maximum difference values
        # top_10_indices = np.argsort(difference)[-10:][::-1]

        # # Get the corresponding wavelengths
        # top_10_wavelengths = observed_planet.wavelength[top_10_indices]

        # print(f"Top 10 wavelengths with maximum difference for bandwidth {bandwidth}:")
        # for w, d in zip(top_10_wavelengths, difference[top_10_indices]):
        #     print(f"Wavelength: {w:.2f} μm, Difference: {d:.2f}")

        axs[1, i].step(
            observed_planet.wavelength, difference, color="black", where="mid"
        )

        for bandpass in bandpasses:
            axs[1, i].axvspan(bandpass[0], bandpass[1], alpha=0.2, color="gray")

        axs[1, i].axvspan(0.45, 0.55, alpha=0.2, color="pink")

        axs[1, i].axhline(y=3, color="gray", linestyle=":", linewidth=1)
        axs[1, i].set_xlabel("Wavelength (μm)", fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"])
        if i == 0:
            axs[1, i].set_ylabel(
                r"$\frac{|F_{Earth}-F_{Neptune}|}{\sigma}$",
                fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
            )
        axs[1, i].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )

    for i in range(3):
        axs[1, i].set_ylim(0, 6)

    fig.suptitle(title, fontsize=FULL_WIDTH_SIZES["TITLE_SIZE"])
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.2)
    return fig, output


def max_snr_finder(
    wavelength,
    difference,
    bandwidth,
    channels=[(0.2, 0.4), (0.4, 1.0), (1.0, 1.8)],
    detection_band=[0.5 * (1 - 0.1), 0.5 * (1 + 0.1)],
):
    band_mask = np.argmin(np.abs(wavelength - 0.5))
    discovery_band_snr = difference[band_mask]
    if bandwidth==0.2:
        print(f"Discovery Bandpass: Band significance = {discovery_band_snr:.3f}")
    bandpasses = []
    lambda_maxdiffs = []
    max_differences = []

    print(f"Bandwidth:{bandwidth}")
    for channel_start, channel_end in channels:
        channel_mask = (wavelength >= channel_start) & (wavelength <= channel_end)
        channel_wavelengths = wavelength[channel_mask]
        channel_difference = difference[channel_mask]

        max_difference = 0
        lambda_maxdiff = None
        bandpass = None
        for lambda_center in channel_wavelengths:
            lower_bound = lambda_center * (1 - bandwidth / 2)
            upper_bound = lambda_center * (1 + bandwidth / 2)

            if (
                lower_bound >= channel_start
                and upper_bound <= channel_end
                and not (detection_band[0] <= lambda_center <= detection_band[1])
                and not (detection_band[0] <= lower_bound <= detection_band[1])
                and not (detection_band[0] <= upper_bound <= detection_band[1])
            ):
                current_difference = channel_difference[
                    channel_wavelengths == lambda_center
                ][0]
                if current_difference > max_difference:
                    max_difference = current_difference
                    lambda_maxdiff = lambda_center
                    bandpass = [lower_bound, upper_bound]

        bandpasses.append(bandpass)
        lambda_maxdiffs.append(lambda_maxdiff)
        max_differences.append(max_difference)
        if lambda_maxdiff is not None:
            print(
                f"Channel {channel_start}-{channel_end}: λ_maxsignificance = {lambda_maxdiff:.3f} μm, Max significance = {max_difference:.2f}"
            )
        else:
            print(
                f"Channel {channel_start}-{channel_end}: No valid λ_maxdifference found"
            )
    return lambda_maxdiffs, max_differences, bandpasses


def color_color_calculator(dataframe, filters):
    idx_1 = np.argmin(np.abs(dataframe["wavelength"] - filters[0]))
    idx_2 = np.argmin(np.abs(dataframe["wavelength"] - filters[1]))
    idx_3 = np.argmin(np.abs(dataframe["wavelength"] - filters[2]))

    # assuming filter is delta function at lambda (bandpass already taken into account)
    f1int = dataframe["flux"][idx_1]
    sigma_f1 = dataframe["flux"][idx_1] / dataframe["snr"][idx_1]
    f2int = dataframe["flux"][idx_2]
    sigma_f2 = dataframe["flux"][idx_2] / dataframe["snr"][idx_2]
    f3int = dataframe["flux"][idx_3]
    sigma_f3 = dataframe["flux"][idx_3] / dataframe["snr"][idx_3]

    f1_f2 = f1int - f2int
    f3_f2 = f3int - f2int

    # f1-f2
    f1_f2_error = np.sqrt((sigma_f1) ** 2 + (sigma_f2) ** 2)

    # f2-f3
    f3_f2_error = np.sqrt((sigma_f3) ** 2 + (sigma_f2) ** 2)
    return f1_f2, f3_f2, f1_f2_error, f3_f2_error


def color_color_calculator_ratio(dataframe, filters):
    idx_1 = np.argmin(np.abs(dataframe["wavelength"] - filters[0]))
    idx_2 = np.argmin(np.abs(dataframe["wavelength"] - filters[1]))
    idx_3 = np.argmin(np.abs(dataframe["wavelength"] - filters[2]))

    # assuming filter is delta function at lambda (bandpass already taken into account)
    f1int = dataframe["flux"][idx_1]
    sigma_f1 = dataframe["flux"][idx_1] / dataframe["snr"][idx_1]
    f2int = dataframe["flux"][idx_2]
    sigma_f2 = dataframe["flux"][idx_2] / dataframe["snr"][idx_2]
    f3int = dataframe["flux"][idx_3]
    sigma_f3 = dataframe["flux"][idx_3] / dataframe["snr"][idx_3]
    
    f1_f2 = f1int / f2int
    f3_f2 = f3int / f2int

    # f1/f2
    f1_f2_error = np.sqrt(
        (1 / f2int * (sigma_f1)) ** 2 + (-(f1int / f2int**2) * (sigma_f2)) ** 2
    )

    # f3/f2
    f3_f2_error = np.sqrt(
        (1 / f2int * (sigma_f3)) ** 2 + (-(f3int / f2int**2) * (sigma_f2)) ** 2
    )
    return f1_f2, f3_f2, f1_f2_error, f3_f2_error


def color_color_calculator_magnitudes(dataframe, filters):
    idx_1 = np.argmin(np.abs(dataframe["wavelength"] - filters[0]))
    idx_2 = np.argmin(np.abs(dataframe["wavelength"] - filters[1]))
    idx_3 = np.argmin(np.abs(dataframe["wavelength"] - filters[2]))

    # assuming filter is delta function at lambda (bandpass already taken into account)
    f1int = dataframe["flux"][idx_1]
    sigma_f1 = dataframe["flux"][idx_1] / dataframe["snr"][idx_1]
    f2int = dataframe["flux"][idx_2]
    sigma_f2 = dataframe["flux"][idx_2] / dataframe["snr"][idx_2]
    f3int = dataframe["flux"][idx_3]
    sigma_f3 = dataframe["flux"][idx_3] / dataframe["snr"][idx_3]

    f1_f2 = -2.5 * np.log10(f1int / f2int)
    f3_f2 = -2.5 * np.log10(f3int / f2int)

    # f1-f2
    # log10(g)=d/dg(log10(g))*d(g) where g = f1/f2 --> 1/(ln(10)*g)*d(g)--> 1/ln(10)*f2/f1 d(f1/f2)
    # derivative over f1 ==> 1/ln(10)*f2/f1*1/f2 = 1/ln(10)/f1 (modulo -2.5)
    # derivative over f2 ==> 1/ln(10)*f2/f1*f1* (-1/f2**2) = -1/ln(10)/f2 (modulo -2.5)
    deriv_over_f1 = -2.5 * (1 / np.log(10) / f1int)
    deriv_over_f2 = -2.5 * (-1 / np.log(10) / f2int)
    f1_f2_error = np.sqrt(
        (deriv_over_f1 * sigma_f1) ** 2 + (deriv_over_f2 * sigma_f2) ** 2
    )

    # f2-f3
    # log10(g)=d/dg(log10(g))*d(g) where g = f3/f2 --> 1/(ln(10)*g)*d(g)--> 1/ln(10)*f2/f3 d(f3/f2)

    # derivative over f3 ==> 1/ln(10)*f2/f3*1/f2 = 1/ln(10)/f3 (modulo -2.5)
    # derivative over f2 ==> 1/ln(10)*f2/f3*f3* (-1/f2**2) = -1/ln(10)/f2 (modulo -2.5)
    deriv_over_f3 = -2.5 * (1 / np.log(10) / f3int)
    deriv_over_f2 = -2.5 * (-1 / np.log(10) / f2int)
    f3_f2_error = np.sqrt(
        (deriv_over_f2 * sigma_f2) ** 2 + (deriv_over_f3 * sigma_f3) ** 2
    )
    return f1_f2, f3_f2, f1_f2_error, f3_f2_error


def color_color_plots_ratio(earth, cold_neptune, warm_neptune, filters, lims=None, lims_offsets=None,plots=True):
    if plots:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
        axs = axs.flatten()
        cold_colormap, warm_colormap, full_colormap = get_colormaps()
        colors = {
            "Earth": "black",
            "Cold Neptune": cold_colormap(0.6),
            "Warm Neptune": warm_colormap(0.4),
        }
        markers = {"Earth": "o", "Cold Neptune": "s", "Warm Neptune": "d"}
    counter = 0
    latex_table = ""
    output={}
    for bw in [0.05, 0.1, 0.2]:
        f1_f2_earth, f3_f2_earth, f1_f2_error_earth, f3_f2_error_earth = (
            color_color_calculator_ratio(earth[bw], filters[bw])
        )

        (
            f1_f2_cold_neptune,
            f3_f2_cold_neptune,
            f1_f2_error_cold_neptune,
            f3_f2_error_cold_neptune,
        ) = color_color_calculator_ratio(cold_neptune[bw], filters[bw])

        (
            f1_f2_warm_neptune,
            f3_f2_warm_neptune,
            f1_f2_error_warm_neptune,
            f3_f2_error_warm_neptune,
        ) = color_color_calculator_ratio(warm_neptune[bw], filters[bw])

        distance_cold, uncertainty_cold = distance_with_uncertainty(
            f3_f2_earth,
            f1_f2_earth,
            f3_f2_cold_neptune,
            f1_f2_cold_neptune,
            f3_f2_error_earth,
            f1_f2_error_earth,
            f3_f2_error_cold_neptune,
            f1_f2_error_cold_neptune,
        )

        distance_warm, uncertainty_warm = distance_with_uncertainty(
            f3_f2_earth,
            f1_f2_earth,
            f3_f2_warm_neptune,
            f1_f2_warm_neptune,
            f3_f2_error_earth,
            f1_f2_error_earth,
            f3_f2_error_warm_neptune,
            f1_f2_error_warm_neptune,
        )
        
        print(
            f"Earth vs cold neptune {bw}: {distance_cold} ± {uncertainty_cold}. Ratio: {distance_cold/uncertainty_cold}"
        )

        print(
            f"Earth vs warm neptune {bw}: {distance_warm} ± {uncertainty_warm}. Ratio: {distance_warm/uncertainty_warm}"
        )
        output[bw]=[distance_cold/uncertainty_cold,distance_warm/uncertainty_warm]
        if plots:
            axs[counter].text(0.95, 0.95, f"E-WN={output[bw][1]:.2f}", color=colors['Warm Neptune'], ha='right', va='top', transform=axs[counter].transAxes,fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
            axs[counter].text(0.95, 0.85, f"E-CN={output[bw][0]:.2f}", color=colors['Cold Neptune'], ha='right', va='top', transform=axs[counter].transAxes,fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
            # Add to LaTeX table
            latex_table += f"& \\multirow{{2}}{{*}}{{{bw}}} & Earth vs cold neptune & {distance_cold/uncertainty_cold:.3f}$\sigma$ \\\\\n"
            latex_table += f" &  & Earth vs warm neptune & {distance_warm/uncertainty_warm:.3f}$\sigma$ \\\\\n"
            if bw != 0.2:
                latex_table += "\\cline{2-4}\n"

            axs[counter].set_title(
                f"Bandwidth: {int(bw*100)}%", fontsize=FULL_WIDTH_SIZES['SUBTITLE_SIZE']
                # f"Bandpass: {bw}. Filters: {[round(f, 2) for f in filters[bw]]}"
            )

            axs[counter].errorbar(
                f3_f2_earth,
                f1_f2_earth,
                xerr=f3_f2_error_earth,
                yerr=f1_f2_error_earth,
                c=colors["Earth"],
                marker=markers["Earth"],
            )
            axs[counter].errorbar(
                f3_f2_cold_neptune,
                f1_f2_cold_neptune,
                xerr=f3_f2_error_cold_neptune,
                yerr=f1_f2_error_cold_neptune,
                c=colors["Cold Neptune"],
                marker=markers["Cold Neptune"],
            )
            axs[counter].errorbar(
                f3_f2_warm_neptune,
                f1_f2_warm_neptune,
                xerr=f3_f2_error_warm_neptune,
                yerr=f1_f2_error_warm_neptune,
                c=colors["Warm Neptune"],
                marker=markers["Warm Neptune"],
            )

            axs[counter].set_ylabel(
                "$F_{"
                + str(np.round(filters[bw][0], 2))
                + "}/F_{"
                + str(np.round(filters[bw][1], 2))
                + "}$",
                fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
            )
            axs[counter].set_xlabel(
                "$F_{"
                + str(np.round(filters[bw][2], 2))
                + "}/F_{"
                + str(np.round(filters[bw][1], 2))
                + "}$",
                fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
            )
            if lims is not None:
                axs[counter].set_xlim(lims[0][0], lims[0][1])
                axs[counter].set_ylim(lims[1][0], lims[1][1])
            elif lims_offsets is not None:
                avg_x = np.average(
                    [
                        f3_f2_earth,
                        f3_f2_cold_neptune,
                        f3_f2_warm_neptune,
                    ]
                )
                avg_y = np.average(
                    [
                        f1_f2_earth,
                        f1_f2_cold_neptune,
                        f1_f2_warm_neptune,
                    ]
                )
                axs[counter].set_xlim(avg_x - lims_offsets[0], avg_x + lims_offsets[0])
                axs[counter].set_ylim(avg_y - lims_offsets[1], avg_y + lims_offsets[1])
        counter += 1
    if plots:
        latex_table += "\\hline\n"
        for ax in axs:
            ax.tick_params(
                    axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
                )
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        plt.tight_layout()

        return fig, latex_table,output
    else:
        return output






def color_color_plots_magnitudes(earth, cold_neptune, warm_neptune, filters, lims=None, lims_offsets=None,plots=True):
    if plots:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
        axs = axs.flatten()
        cold_colormap, warm_colormap, full_colormap = get_colormaps()
        colors = {
            "Earth": "black",
            "Cold Neptune": cold_colormap(0.6),
            "Warm Neptune": warm_colormap(0.4),
        }
        markers = {"Earth": "o", "Cold Neptune": "s", "Warm Neptune": "d"}
    counter = 0
    latex_table = ""
    output={}
    for bw in [0.05, 0.1, 0.2]:
        f1_f2_earth, f3_f2_earth, f1_f2_error_earth, f3_f2_error_earth = (
            color_color_calculator_magnitudes(earth[bw], filters[bw])
        )
        (
            f1_f2_cold_neptune,
            f3_f2_cold_neptune,
            f1_f2_error_cold_neptune,
            f3_f2_error_cold_neptune,
        ) = color_color_calculator_magnitudes(cold_neptune[bw], filters[bw])
        (
            f1_f2_warm_neptune,
            f3_f2_warm_neptune,
            f1_f2_error_warm_neptune,
            f3_f2_error_warm_neptune,
        ) = color_color_calculator_magnitudes(warm_neptune[bw], filters[bw])

        distance_cold, uncertainty_cold = distance_with_uncertainty(
            f3_f2_earth,
            f1_f2_earth,
            f3_f2_cold_neptune,
            f1_f2_cold_neptune,
            f3_f2_error_earth,
            f1_f2_error_earth,
            f3_f2_error_cold_neptune,
            f1_f2_error_cold_neptune,
        )

        distance_warm, uncertainty_warm = distance_with_uncertainty(
            f3_f2_earth,
            f1_f2_earth,
            f3_f2_warm_neptune,
            f1_f2_warm_neptune,
            f3_f2_error_earth,
            f1_f2_error_earth,
            f3_f2_error_warm_neptune,
            f1_f2_error_warm_neptune,
        )
        print(
            f"Earth vs cold neptune {bw}: {distance_cold} ± {uncertainty_cold}. Ratio: {distance_cold/uncertainty_cold}"
        )

        print(
            f"Earth vs warm neptune {bw}: {distance_warm} ± {uncertainty_warm}. Ratio: {distance_warm/uncertainty_warm}"
        )
        output[bw]=[distance_cold/uncertainty_cold,distance_warm/uncertainty_warm]
        if plots:
            axs[counter].text(0.95, 0.95, f"E-WN={output[bw][1]:.2f}", color=colors['Warm Neptune'], ha='right', va='top', transform=axs[counter].transAxes,fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
            axs[counter].text(0.95, 0.85, f"E-CN={output[bw][0]:.2f}", color=colors['Cold Neptune'], ha='right', va='top', transform=axs[counter].transAxes,fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
            # Add to LaTeX table
            latex_table += f"& \\multirow{{2}}{{*}}{{{bw}}} & Earth vs cold neptune & {distance_cold/uncertainty_cold:.3f}$\sigma$ \\\\\n"
            latex_table += f" &  & Earth vs warm neptune & {distance_warm/uncertainty_warm:.3f}$\sigma$ \\\\\n"
            if bw != 0.2:
                latex_table += "\\cline{2-4}\n"

            axs[counter].set_title(
                f"Bandwidth: {int(bw*100)}%", fontsize=FULL_WIDTH_SIZES['SUBTITLE_SIZE']+5
                # f"Bandpass: {bw}. Filters: {[round(f, 2) for f in filters[bw]]}"
            )

            axs[counter].errorbar(
                f3_f2_earth,
                f1_f2_earth,
                xerr=f3_f2_error_earth,
                yerr=f1_f2_error_earth,
                c=colors["Earth"],
                marker=markers["Earth"],
            )
            axs[counter].errorbar(
                f3_f2_cold_neptune,
                f1_f2_cold_neptune,
                xerr=f3_f2_error_cold_neptune,
                yerr=f1_f2_error_cold_neptune,
                c=colors["Cold Neptune"],
                marker=markers["Cold Neptune"],
            )
            axs[counter].errorbar(
                f3_f2_warm_neptune,
                f1_f2_warm_neptune,
                xerr=f3_f2_error_warm_neptune,
                yerr=f1_f2_error_warm_neptune,
                c=colors["Warm Neptune"],
                marker=markers["Warm Neptune"],
            )

            axs[counter].set_ylabel(
                "$mag_{"
                + str(np.round(filters[bw][0], 2))
                + "}-mag_{"
                + str(np.round(filters[bw][1], 2))
                + "}$",
                fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
            )
            axs[counter].set_xlabel(
                "$mag_{"
                + str(np.round(filters[bw][2], 2))
                + "}-mag_{"
                + str(np.round(filters[bw][1], 2))
                + "}$",
                fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
            )
            if lims is not None:
                axs[counter].set_xlim(lims[0][0], lims[0][1])
                axs[counter].set_ylim(lims[1][0], lims[1][1])
            elif lims_offsets is not None:
                avg_x = np.average(
                    [
                        f3_f2_earth,
                        f3_f2_cold_neptune,
                        f3_f2_warm_neptune,
                    ]
                )
                avg_y = np.average(
                    [
                        f1_f2_earth,
                        f1_f2_cold_neptune,
                        f1_f2_warm_neptune,
                    ]
                )
                axs[counter].set_xlim(avg_x - lims_offsets[0], avg_x + lims_offsets[0])
                axs[counter].set_ylim(avg_y - lims_offsets[1], avg_y + lims_offsets[1])
        counter += 1
    if plots:
        latex_table += "\\hline\n"

        axs[0].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )
        axs[1].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )
        axs[2].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )

        plt.tight_layout()

        return fig, latex_table,output
    else:
        return output


def color_color_plots(
    earth, cold_neptune, warm_neptune, filters, lims=None, lims_offsets=None,plots=True
):
    if plots:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
        axs = axs.flatten()
        cold_colormap, warm_colormap, full_colormap = get_colormaps()
        colors = {
            "Earth": "black",
            "Cold Neptune": cold_colormap(0.6),
            "Warm Neptune": warm_colormap(0.4),
        }
        markers = {"Earth": "o", "Cold Neptune": "s", "Warm Neptune": "d"}
    counter = 0
    latex_table = ""
    output={}
    for bw in [0.05, 0.1, 0.2]:
        f1_f2_earth, f3_f2_earth, f1_f2_error_earth, f3_f2_error_earth = (
            color_color_calculator(earth[bw], filters[bw])
        )
        (
            f1_f2_cold_neptune,
            f3_f2_cold_neptune,
            f1_f2_error_cold_neptune,
            f3_f2_error_cold_neptune,
        ) = color_color_calculator(cold_neptune[bw], filters[bw])
        (
            f1_f2_warm_neptune,
            f3_f2_warm_neptune,
            f1_f2_error_warm_neptune,
            f3_f2_error_warm_neptune,
        ) = color_color_calculator(warm_neptune[bw], filters[bw])

        distance_cold, uncertainty_cold = distance_with_uncertainty(
            f3_f2_earth,
            f1_f2_earth,
            f3_f2_cold_neptune,
            f1_f2_cold_neptune,
            f3_f2_error_earth,
            f1_f2_error_earth,
            f3_f2_error_cold_neptune,
            f1_f2_error_cold_neptune,
        )

        distance_warm, uncertainty_warm = distance_with_uncertainty(
            f3_f2_earth,
            f1_f2_earth,
            f3_f2_warm_neptune,
            f1_f2_warm_neptune,
            f3_f2_error_earth,
            f1_f2_error_earth,
            f3_f2_error_warm_neptune,
            f1_f2_error_warm_neptune,
        )
        print(
            f"Earth vs cold neptune {bw}: {distance_cold} ± {uncertainty_cold}. Ratio: {distance_cold/uncertainty_cold}"
        )

        print(
            f"Earth vs warm neptune {bw}: {distance_warm} ± {uncertainty_warm}. Ratio: {distance_warm/uncertainty_warm}"
        )
        output[bw]=[distance_cold/uncertainty_cold,distance_warm/uncertainty_warm]
        if plots:
            axs[counter].text(0.95, 0.95, f"E-WN={output[bw][1]:.2f}", color=colors['Warm Neptune'], ha='right', va='top', transform=axs[counter].transAxes,fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
            axs[counter].text(0.95, 0.85, f"E-CN={output[bw][0]:.2f}", color=colors['Cold Neptune'], ha='right', va='top', transform=axs[counter].transAxes,fontsize=FULL_WIDTH_SIZES['LEGEND_SIZE'])
            # Add to LaTeX table
            latex_table += f"& \\multirow{{2}}{{*}}{{{bw}}} & Earth vs cold neptune & {distance_cold/uncertainty_cold:.3f}$\sigma$ \\\\\n"
            latex_table += f" &  & Earth vs warm neptune & {distance_warm/uncertainty_warm:.3f}$\sigma$ \\\\\n"
            if bw != 0.2:
                latex_table += "\\cline{2-4}\n"

            axs[counter].set_title(
                f"Bandwidth: {int(bw*100)}%", fontsize=FULL_WIDTH_SIZES['SUBTITLE_SIZE']+5
                # f"Bandpass: {bw}. Filters: {[round(f, 2) for f in filters[bw]]}"
            )

            axs[counter].errorbar(
                f3_f2_earth,
                f1_f2_earth,
                xerr=f3_f2_error_earth,
                yerr=f1_f2_error_earth,
                c=colors["Earth"],
                marker=markers["Earth"],
            )
            axs[counter].errorbar(
                f3_f2_cold_neptune,
                f1_f2_cold_neptune,
                xerr=f3_f2_error_cold_neptune,
                yerr=f1_f2_error_cold_neptune,
                c=colors["Cold Neptune"],
                marker=markers["Cold Neptune"],
            )
            axs[counter].errorbar(
                f3_f2_warm_neptune,
                f1_f2_warm_neptune,
                xerr=f3_f2_error_warm_neptune,
                yerr=f1_f2_error_warm_neptune,
                c=colors["Warm Neptune"],
                marker=markers["Warm Neptune"],
            )

            axs[counter].set_ylabel(
                "$F_{"
                + str(np.round(filters[bw][0], 2))
                + "}-F_{"
                + str(np.round(filters[bw][1], 2))
                + "}$",
                fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
            )
            axs[counter].set_xlabel(
                "$F_{"
                + str(np.round(filters[bw][2], 2))
                + "}-F_{"
                + str(np.round(filters[bw][1], 2))
                + "}$",
                fontsize=FULL_WIDTH_SIZES["LABEL_SIZE"],
            )
            if lims is not None:
                axs[counter].set_xlim(lims[0][0], lims[0][1])
                axs[counter].set_ylim(lims[1][0], lims[1][1])
            elif lims_offsets is not None:
                avg_x = np.average(
                    [
                        f3_f2_earth,
                        f3_f2_cold_neptune,
                        f3_f2_warm_neptune,
                    ]
                )
                avg_y = np.average(
                    [
                        f1_f2_earth,
                        f1_f2_cold_neptune,
                        f1_f2_warm_neptune,
                    ]
                )
                axs[counter].set_xlim(avg_x - lims_offsets[0], avg_x + lims_offsets[0])
                axs[counter].set_ylim(avg_y - lims_offsets[1], avg_y + lims_offsets[1])
        counter += 1
    if plots:
        latex_table += "\\hline\n"

        axs[0].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )
        axs[1].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )
        axs[2].tick_params(
            axis="both", which="major", labelsize=FULL_WIDTH_SIZES["TICK_SIZE"]
        )

        plt.tight_layout()

        return fig, latex_table,output
    else:
        return output


def distance_with_uncertainty(x1, y1, x2, y2, sigma_x1, sigma_y1, sigma_x2, sigma_y2):
    # Calculate distance
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Calculate uncertainty
    # reminder, derivative of sqrt is 1/(2*sqrt) hence derivative of d becomes 1/(2*d)
    # derivative over x1 ==> 1/2/d *(2*(x2-x1)*(-1)) == 1/d*(-(x2-x1))
    # derivative over x2 ==> 1/2/d *(2*(x2-x1)) == 1/d*((x2-x1))
    # derivative over y1 ==> 1/2/d * (2*(y2-y1)*(-1))== 1/d*(-(y2-y1))
    # derivative over y2 ==> 1/2/d * (2*(y2-y1)) == 1/d*(y2-y1)
    deriv_over_x1 = -(x2 - x1) / d
    deriv_over_x2 = (x2 - x1) / d
    deriv_over_y1 = -(y2 - y1) / d
    deriv_over_y2 = (y2 - y1) / d
    sigma_d = np.sqrt(
        (deriv_over_x1 * sigma_x1) ** 2
        + (deriv_over_x2 * sigma_x2) ** 2
        + (deriv_over_y1 * sigma_y1) ** 2
        + (deriv_over_y2 * sigma_y2) ** 2
    )

    return d, sigma_d



def distance_with_uncertainty_JKT16(xearth, yearth, xneptune, yneptune, sigma_xearth, sigma_yearth, sigma_xneptune, sigma_yneptune):
    # Calculate distance

    x=(xearth - xneptune) 
    y=(yearth - yneptune) 
    d = np.sqrt(x ** 2 +y ** 2)

    sigma_d=(x**2+y**2)/np.sqrt(x**2*xearth*sigma_xearth+y**2*yearth*sigma_yearth)
    return d, sigma_d
