from core import Node, as_function_node
from ase.atoms import Atoms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def birch_murnaghan(vol, E0, V0, B0, BP):
    """
    Birch-Murnaghan EOS.
    """
    E = E0 + (9.0*V0*B0)/16.0 * ( ((V0/vol)**(2.0/3.0)-1.0)**3.0 *BP +
        ((V0/vol)**(2.0/3.0)-1.0)**2.0 * (6.0-4.0*(V0/vol)**(2.0/3.0)))
    return E


@as_function_node
def FitBirchMurnaghanEOS(ev_curve_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Fits the energy vs volume data to the Birch-Murnaghan EOS
    and returns a tuple of equilibrium properties: (E0, V0, B0).
    """
    from scipy.optimize import curve_fit

    volumes = ev_curve_df["volume"].values
    energies = ev_curve_df["energy"].values

    # Initial guesses for the fitting parameters
    V0_guess = volumes[np.argmin(energies)]
    E0_guess = min(energies)
    B0_guess = 1.0  # in eV/Å³
    B1_guess = 4.0

    popt, _ = curve_fit(birch_murnaghan, volumes, energies, p0=[E0_guess, V0_guess, B0_guess, B1_guess])
    E0, V0, B0, B1 = popt
    B0_GPa = B0 * 160.21766208  # Conversion factor

    return E0, V0, B0_GPa


@as_function_node()
def PlotEVCurve(
    ev_curve_df: pd.DataFrame,
    xlabel: str = "Volume (Å³)",
    ylabel: str = "Energy (eV)",
    title: str = "Energy vs Volume Curve",
    fontsize: int = 12
):
    """
    Plots the Energy vs. Volume (EV) curve from a computed EV dataset.

    Args:
        ev_curve_df (pd.DataFrame): DataFrame containing 'volume' and 'energy' columns.
        xlabel (str, optional): Label for the x-axis. Defaults to "Volume (Å³)".
        ylabel (str, optional): Label for the y-axis. Defaults to "Energy (eV)".
        title (str, optional): Title of the plot. Defaults to "Energy vs Volume Curve".
        fontsize (int, optional): Font size for labels and title. Defaults to 12.

    Returns:
        fig, ax: The matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots()
    ax.plot(ev_curve_df['volume'], ev_curve_df['energy'], marker='o', linestyle='-', color='b')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    # ax.legend()
    return fig
