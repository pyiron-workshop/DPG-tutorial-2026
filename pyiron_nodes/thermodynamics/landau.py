# from __future__ import annotations

from typing import Iterable, Optional, Tuple

import landau
from landau.phases import Phase, LinePhase, TemperatureDependentLinePhase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core import as_function_node

@as_function_node
def LinePhase(
    name: str, 
    fixed_concentration: float, 
    line_energy: float, 
    line_entropy: float
) -> LinePhase:
    """
    Create a temperature-independent *line compound* phase at a fixed composition.

    **Scientific purpose**
    Represent a stoichiometric (single-composition) phase with a simple linear free-energy
    model, typically
    \n    f(T) = E - T S
    \nwhere `E` is the energy/intercept and `S` is the entropy/slope. This is commonly used
    for terminal reference states and line compounds in CALPHAD-/Landau-style phase-diagram
    calculations.

    **Required inputs**
    - ``name``: phase label used in tables and plots.
    - ``fixed_concentration``: composition (mole fraction) at which this phase exists (usually in [0, 1]).
    - ``line_energy``: energy/intercept term (e.g., eV/atom).
    - ``line_entropy``: entropy/slope term (e.g., eV/atom/K), contributing ``-T*line_entropy``.

    **Typical use-cases**
    * Define stoichiometric phases (line compounds) for binary phase diagrams.
    * Provide endmember phases for building solution models (ideal/regular solutions).
    * Quick two-phase comparisons when only linear-in-T free energies are available.

    Returns
    -------
    A ``landau.phases.LinePhase`` instance compatible with ``landau.calculate.calc_phase_diagram``.
    """
    from landau.phases import LinePhase

    phase = LinePhase(
        name=name,
        fixed_concentration=fixed_concentration, 
        line_energy=line_energy, 
        line_entropy=line_entropy
    )
    
    return phase

@as_function_node
def TemperatureDependentLinePhase(
    name: str,
    fixed_concentration: float,
    temperatures: np.ndarray | list[float],
    free_energies: np.ndarray | list[float],
    num_parameters: int | None = 3
) -> TemperatureDependentLinePhase:
    """
    Create a fixed-composition phase with *temperature-dependent* free energy from tabulated data.

    **Scientific purpose**
    Fit/interpolate sampled free energies f(T) at one composition into a smooth model using an
    SGTE-style parameterization. This is the standard way to turn discrete thermodynamic data
    (e.g., from CALPHAD assessments, DFT+phonons, MD free energies) into a phase object usable
    in phase-diagram calculations.

    **Required inputs**
    - ``name``: phase label used in tables and plots.
    - ``fixed_concentration``: composition (mole fraction) at which the data applies.
    - ``temperatures``: temperature samples (K).
    - ``free_energies``: free energies (same length as ``temperatures``, typically eV/atom).
    - ``num_parameters``: number of SGTE coefficients used to fit/interpolate f(T).

    **Typical use-cases**
    * Use computed/assessed f(T) data for a stoichiometric compound or a phase at a single composition.
    * Later combine multiple such line phases across compositions into a solution model.

    Returns
    -------
    A ``landau.phases.TemperatureDependentLinePhase`` with an SGTE interpolator, usable in
    ``CalcPhaseDiagram`` / ``landau.calculate.calc_phase_diagram``.
    """
    import numpy as np
    from landau.interpolate import SGTE
    from landau.phases import TemperatureDependentLinePhase
    
    interpolator=SGTE(num_parameters)

    phase = TemperatureDependentLinePhase(
        name=name, 
        fixed_concentration=fixed_concentration, 
        temperatures=temperatures, 
        free_energies=free_energies,
        interpolator=interpolator
    )
    
    return phase

@as_function_node
def TransitionTemperature(
    phase1: Phase, 
    phase2: Phase,
    Tmin: float,
    Tmax: float,
) -> plt.Figure:
    """
    Plot free energy vs temperature for two phases and mark their crossing (transition temperature).

    **Scientific purpose**
    Identify a phase transition temperature (e.g., polymorph A→B) by locating where the two
    free-energy curves intersect. This is a common analysis for line compounds or fixed-composition
    phases where the equilibrium transition is well approximated by the f(T) crossing.

    **Required inputs**
    - ``phase1``, ``phase2``: two ``landau`` Phase objects to compare.
    - ``Tmin``, ``Tmax``: temperature search window (K).

    **Assumptions / limitations**
    This routine computes stability at ``mu = 0`` and is most meaningful when both phases refer
    to the *same concentration*. If the phases have different concentrations, the reported crossing
    may not correspond to true equilibrium.

    **Typical use-cases**
    * Determine the melting/ordering/polymorphic transition temperature at fixed composition.
    * Sanity-check fitted/interpolated f(T) models by visual inspection.

    Returns
    -------
    A Matplotlib ``Figure`` showing both free-energy curves and an annotated transition temperature.
    """
    from landau.phases import Phase
    from landau.calculate import calc_phase_diagram
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    df = calc_phase_diagram(
        phases=[phase1, phase2], 
        Ts=np.linspace(Tmin, Tmax), 
        mu=0.0,
        refine=True,
        keep_unstable=True
    )

    try:
        fm, Tm = df.query('border and T!=@Tmin and T!=@Tmax')[['f','T']].iloc[0].tolist()
    except IndexError:
        print("Transition Point not found!")
        fm, Tm = np.nan, np.nan

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x='T', y='f',
        hue='phase',
        style='stable', style_order=[True, False],
        ax=ax,
    )
    ax.axvline(Tm, color='k', linestyle='dotted', alpha=.5)
    ax.scatter(Tm, fm, marker='o', c='k', zorder=10)

    dfa = np.ptp(df['f'].dropna())
    dft = np.ptp(df['T'].dropna())
    ax.text(Tm + .05 * dft, fm + dfa * .1, rf"$T_m = {Tm:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Free Energy [eV/atom]")

    return fig

@as_function_node
def IdealSolution(
    name: str, 
    phase1: Phase, 
    phase2: Phase
) -> Phase:
    """
    Build an *ideal mixing* solution phase from two endmember phases.

    **Scientific purpose**
    Create a binary solution model with ideal configurational entropy (no excess enthalpy),
    commonly used as a baseline mixing model between two terminal phases.

    **Required inputs**
    - ``name``: name of the resulting solution phase.
    - ``phase1``, ``phase2``: endmember phases (often line phases near c=0 and c=1).

    **Typical use-cases**
    * Model substitutional solid solutions without interaction parameters.
    * Provide a reference solution to compare against regular/non-ideal solution models.

    Returns
    -------
    A ``landau.phases.IdealSolution`` (returned as ``Phase``) suitable for phase-diagram calculations.
    """
    from landau.phases import IdealSolution

    solution_phase = IdealSolution(
        name=name, 
        phase1=phase1, 
        phase2=phase2
    )

    return solution_phase

def guess_mu_range(
    phases: Iterable[Phase], 
    T: float, 
    samples: int
) -> np.ndarray:
    """Guess chemical potential window from the ideal solution.

    Searches numerically for chemical potentials which stabilize
    concentrations close to 0 and 1 and then use the concentrations
    encountered along the way to numerically invert the c(mu) mapping.
    Using an even c grid with mu(c) then yields a decent sampling of mu
    space so that the final phase diagram is described everywhere equally.

    Args:
        phases: list of phases to consider
        T: temperature at which to estimate chemical potential range
        samples: how many mu samples to return

    Returns:
        array of chemical potentials that likely cover the whole concentration space
    """

    import scipy.interpolate as si
    import numpy as np
    # semigrand canonical "average" concentration
    # use this to avoid discontinuities and be phase agnostic
    def c(mu):
        phis = np.array([p.semigrand_potential(T, mu) for p in phases])
        conc = np.array([p.concentration(T, mu) for p in phases])
        phis -= phis.min()
        beta = 1/(T*8.6e-5)
        prob = np.exp(-beta*phis)
        prob /= prob.sum()
        return (prob * conc).sum()
    cc, mm = [], []
    mu0, mu1 = 0, 0
    while (ci := c(mu0)) > 0.001:
        cc.append(ci)
        mm.append(mu0)
        mu0 -= 0.05
    while (ci := c(mu1)) < 0.999:
        cc.append(ci)
        mm.append(mu1)
        mu1 += 0.05
    cc = np.array(cc)
    mm = np.array(mm)
    I = cc.argsort()
    cc = cc[I]
    mm = mm[I]

    return si.interp1d(cc, mm)(np.linspace(min(cc), max(cc), samples))

@as_function_node
def CalcPhaseDiagram(
    phases: list,
    temperatures: list[float] | np.ndarray,
    chemical_potentials: list[float] | np.ndarray | int = 100,
    refine: bool = True,
    store: bool = True
) -> pd.DataFrame:
    """
    Compute a semigrand-canonical phase-diagram dataset over temperature and chemical potential.

    **Scientific purpose**
    Sample thermodynamic stability across temperature T and chemical potential difference μ to
    determine stable phases, phase boundaries, equilibrium concentrations, and free energies.
    The result is a tabular dataset that can be plotted as c–T (composition–temperature) or μ–T
    diagrams.

    **Required inputs**
    - ``phases``: list of ``landau`` phase objects (line phases and/or solution phases).
    - ``temperatures``: temperature grid (K).
    - ``chemical_potentials``:
        * an explicit μ grid (eV), or
        * an integer N meaning “auto-generate N μ samples”.
      When an integer is given, μ bounds are estimated to approximately span c≈0…1 at the highest T.

    **Typical use-cases**
    * Generate data for binary phase diagrams in c–T or μ–T form.
    * Locate phase boundaries and triple points (optionally with refinement).
    * Produce input for ``PlotConcPhaseDiagram`` and ``PlotMuPhaseDiagram``.

    Returns
    -------
    A ``pandas.DataFrame`` of sampled states (T, μ, c, f, phase labels, stability/border flags),
    as produced by ``landau.calculate.calc_phase_diagram``.
    """
    from landau.calculate import calc_phase_diagram

    if isinstance(chemical_potentials, int):
        mus = guess_mu_range(
            phases=phases, 
            T=max(temperatures),
            samples=chemical_potentials
        )
    else:
        mus = chemical_potentials

    phase_data = calc_phase_diagram(
        phases=phases, 
        Ts=np.asarray(temperatures), 
        mu=mus,
        refine=refine, 
        keep_unstable=False
    )

    return phase_data

@as_function_node
def PlotConcPhaseDiagram(
    phase_data,
    plot_samples: bool = False,
    plot_isolines: bool = False,
    plot_tielines: bool = True,
    linephase_width: float = 0.01,
    concavity: float | None = None,
) -> plt.Figure:
    """
    Plot a concentration–temperature (c–T) phase diagram from a Landau phase-diagram DataFrame.

    **Scientific purpose**
    Visualize phase fields in composition–temperature space for a binary system, highlighting
    single-phase regions and coexistence regions derived from semigrand-canonical sampling.

    **Required inputs**
    - ``phase_data``: DataFrame output from ``CalcPhaseDiagram`` (must contain at least ``c``, ``T``, ``phase``).

    **Optional overlays**
    - ``plot_samples``: show the raw sampled (c, T) points used to build the diagram.
    - ``plot_isolines``: draw lines of constant chemical potential μ (helps diagnose sampling).
    - ``plot_tielines``: draw tie lines at refined triple points (if refinement metadata exists).

    **Plot controls**
    - ``linephase_width``: treat very narrow-solubility phases as rectangles (line-compound styling).
    - ``concavity``: smoothing/shape parameter for polygon fitting; higher is smoother (1≈convex hull).

    Returns
    -------
    A Matplotlib ``Figure`` containing the c–T phase diagram.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import landau
    
    fig, ax = plt.subplots()
    landau.plot.plot_phase_diagram(
            df=phase_data.drop('refined', errors='ignore', axis='columns'),
            min_c_width=linephase_width,
            alpha=concavity or 0.1,
    )
    if plot_samples:
        sns.scatterplot(
            data=phase_data,
            x='c', y='T',
            hue='phase',
            legend=False,
            s=1
        )
    if plot_isolines:
        sns.lineplot(
            data=phase_data.loc[np.isfinite(phase_data.mu)],
            x='c', y='T',
            hue='mu',
            units='phase', estimator=None,
            legend=False,
            sort=False,
        )
    if plot_tielines and 'refined' in phase_data.columns:
        # hasn't made it upstream yet
        for T, dd in phase_data.query('refined=="delaunay-triple"').groupby('T'):
            plt.plot(dd.c, [T]*3, c='k', alpha=.5, zorder=-10)
    plt.xlabel("Concentration")
    plt.ylabel("Temperature [K]")
    
    return fig

@as_function_node
def PlotMuPhaseDiagram(phase_data) -> plt.Figure:
    """
    Plot a chemical potential–temperature (μ–T) phase diagram from a Landau phase-diagram DataFrame.

    **Scientific purpose**
    Visualize which phase is stable as a function of temperature and chemical potential difference μ.
    This representation is natural for semigrand-canonical calculations and is useful for diagnosing
    phase boundaries and stability switches.

    **Required inputs**
    - ``phase_data``: DataFrame output from ``CalcPhaseDiagram`` (should contain ``mu``, ``T``, ``phase``;
      if it contains ``border``, boundary points are highlighted).

    **Typical use-cases**
    * Inspect stability regions in μ–T space.
    * Debug/validate phase-diagram sampling before converting to c–T plots.

    Returns
    -------
    A Matplotlib ``Figure`` containing the μ–T phase diagram with phases colored by label and
    boundaries optionally shown in black.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    border = None
    if 'border' not in phase_data.columns:
        body = phase_data.query('not border')
    else:
        border = phase_data.query('border')
        body = phase_data.query('not border')
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=body,
        x='mu', y='T',
        hue='phase',
        s=5,
        ax=ax
    )
    if border is not None:
        sns.scatterplot(
            data=border,
            x='mu', y='T',
            c='k',
            s=5,
            ax=ax
        )
    ax.set_xlabel("Chemical Potential Difference [eV]")
    ax.set_ylabel("Temperature [K]")
    return fig

@as_function_node
def PhasesFromDataFrame(
    dataframe: pd.DataFrame,
    temperature_parameters: int | None = 4,
    concentration_parameters: int | None = 1,
):
    """
    Convert tabulated free-energy data into Landau phase objects (line phases and solution phases).

    **Scientific purpose**
    Turn a CALPHAD-/DFT-like dataset (free energy vs temperature at multiple compositions) into a set
    of `landau` Phase objects with interpolated temperature dependence (SGTE) and optional composition
    interpolation. This enables automated phase-diagram construction directly from a DataFrame.

    **Required input data columns**
    - ``phase``: phase name; rows with the same name are grouped into one phase model.
    - ``composition``: mole fraction/composition coordinate (typically in [0, 1]).
    - ``temperature``: temperature samples (K) for that composition (often stored as an array per row).
    - ``free_energy``: free energies (eV/atom) corresponding to ``temperature``.

    **How it builds phases**
    1. For each (phase, composition) it builds a temperature-dependent line phase f(T).
    2. If multiple compositions exist:
       - if terminals (c=0 and c=1) only: create an IdealSolution,
       - if terminals plus intermediates: create a RegularSolution,
       - otherwise: create an InterpolatingPhase.
       If ``concentration_parameters is None``, it returns individual line phases instead of a solution.

    **Typical use-cases**
    * Build an entire binary phase model from a table of assessed/computed free energies.
    * Rapidly prototype phase diagrams from high-throughput thermodynamic data.

    Returns
    -------
    ``(phase_list, phase_dict)``
    - ``phase_list``: list of created ``landau.phases.Phase`` objects.
    - ``phase_dict``: dict mapping phase name to object for convenient lookup.
    """
    phases = dataframe.groupby('phase')[dataframe.columns].apply(
            make_phase, include_groups=False,
            temperature_parameters=temperature_parameters,
            concentration_parameters=concentration_parameters,
    )
    phase_dict = {p.name: p for p in phases.explode()}
    print("Found phases:", *phase_dict.keys(), sep='\n')
    phase_list = list(phase_dict.values())
    return phase_list, phase_dict


def make_phase(
    dd: pd.DataFrame, 
    temperature_parameters, 
    concentration_parameters
):  
    from dataclasses import replace

    name = dd.phase.iloc[0]
    # minus 2 for terminals
    # minus 1 to be not exactly interpolating
    sub = [landau.phases.TemperatureDependentLinePhase(
                f'{row.phase}_{c:.03}', c, 
                row.temperature, row.free_energy, 
                interpolator=landau.interpolate.SGTE(temperature_parameters)
            ) for c, row in dd.set_index('composition').iterrows()]
    # only a single concentration
    if len(sub) == 1:
        return replace(sub[0], name=name)
    if concentration_parameters is not None:
        interp_params = min(len(dd)-2-1, concentration_parameters)
        # terminals are present
        if len({0, 1}.intersection([s.line_concentration for s in sub]))==2:
            if len(sub) == 2: # only terminals are present
                return landau.phases.IdealSolution(name, *sub)
            else:
                return landau.phases.RegularSolution(name, sub, interp_params)
        else:
            return landau.phases.InterpolatingPhase(name, sub, interp_params, num_samples=1000)
    else:
        return sub