from ase.atoms import Atoms
from core import as_function_node
import pandas as pd
from typing import Optional 


@as_function_node("plot")
def PlotSpaceGroups(
    df_structures: pd.DataFrame,
    logscale: bool = False,
    figsize_x: int = 12,
    figsize_y: int = 6
):
    """
    Visualize the distribution of crystal space groups in a structure dataset.

    This node generates a histogram of space group numbers for a collection
    of atomistic structures, typically produced by symmetry-based structure
    generation or crystal structure prediction workflows. The plot highlights
    crystallographic systems (triclinic through cubic) using colored background
    spans to aid interpretation.

    The function is intended for exploratory data analysis and validation of
    structure sampling diversity in high-throughput materials simulations.

    Args:
        df_structures (pandas.DataFrame):
            DataFrame containing structural data. Must include a ``"spacegroup"``
            column with integer space group numbers (1–230).
        logscale (bool):
            If ``True``, use a logarithmic scale for the y-axis to emphasize
            rare space groups.
        figsize_x (int):
            Width of the generated figure in inches.
        figsize_y (int):
            Height of the generated figure in inches.

    Returns:
        matplotlib.figure.Figure:
            Matplotlib figure object containing the space group histogram.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'figure.figsize': [8.5,8.5],'axes.labelsize': 16,
                     'xtick.labelsize': 14.5, 'ytick.labelsize': 14.5, 'legend.fontsize': 12,
                     'axes.titlesize': 16})
    spacegroups = df_structures['spacegroup']
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Crystal system ranges
    span_dict = {
        "triclinic": (1, 2),
        "monoclinic": (3, 15),
        "orthorhombic": (16, 74),
        "tetragonal": (75, 142),
        "trigonal": (143, 167),
        "hexagonal": (168, 194),
        "cubic": (195, 230),
    }
    # bar plot
    sg_counts = spacegroups.value_counts().sort_index()
    ax.bar(sg_counts.index, sg_counts.values, width=0.8, color="black", alpha = 0.75)
    if logscale:
        ax.set_yscale("log")
        multiplier = 10
    else:
        multiplier = 1.5

    current_ylim = ax.get_ylim()
    ax.set_ylim(ymax=current_ylim[1] * multiplier)

    xticks = []
    for i, (key, (start, end)) in enumerate(span_dict.items()):
        xticks.append(end)
        ax.axvspan(start, end, alpha=0.35, color=f"C{i}")
        ax.text(
            (start + end) / 2,
            ax.get_ylim()[1] * 0.6,
            key.capitalize(),
            ha="center",
            va="center",
            fontsize=14,
            color=f"C{i}",
            rotation=90,
        )

    ax.set_xlim(0, 230)
    ax.set_xticks(xticks)
    ax.set_xlabel("Space Group")
    ax.set_ylabel("# Structures")
    return fig


@as_function_node
def PlotEnergyVolumeHistogram(
    df_structures: pd.DataFrame,
    gridsize: int = 100,
    shift_convex: bool = False
):
    """
    Plot an energy–volume density map for atomistic structures.

    This node visualizes the distribution of structures in atomic
    energy–volume space using a hexagonal binning histogram. Energies
    and volumes are normalized per atom, making the plot suitable for
    comparing structures of different sizes.

    The visualization is commonly used to:
    - Inspect the energy landscape of generated structures
    - Identify low-energy regions and outliers
    - Assess structure generation quality before relaxation or screening

    Args:
        df_structures (pandas.DataFrame):
            DataFrame containing structure data. Must include the columns
            ``"energy"``, ``"volume"``, and ``"number_of_atoms"``.
        gridsize (int):
            Number of hexagons in the x-direction for the hexbin plot.
            Higher values give finer resolution.
        shift_convex (bool):
            If ``True``, shift energies so that the minimum energy is zero.
            This is useful for convex-hull–like visualizations.

    Returns:
        matplotlib.figure.Figure:
            Matplotlib figure object containing the energy–volume histogram.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    N = df_structures["number_of_atoms"].to_numpy()
    E = df_structures["energy"].to_numpy() / N
    V = df_structures["volume"].to_numpy() / N
    fig, ax = plt.subplots(figsize=(9, 7))

    if shift_convex:
        E = E - E.min()
    df_plot = pd.DataFrame({'V': V, 'E': E})
    hb = ax.hexbin(
        df_plot['V'], 
        df_plot['E'],
        gridsize=gridsize,
        cmap='viridis',
        norm=mcolors.LogNorm(vmin=1)
    )
    plt.colorbar(hb, ax=ax, label='# Structures')

    ax.set_xlabel('Atomic Volume ($\\AA^3$)')
    ax.set_ylabel('Atomic Energy (eV)')
    ax.set_xlim(left=0)
    fig.tight_layout()
    
    return fig