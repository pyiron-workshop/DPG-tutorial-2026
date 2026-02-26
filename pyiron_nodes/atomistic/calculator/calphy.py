from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from ase import Atoms
from calphy.input import Calculation
from core import as_inp_dataclass_node, as_function_node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import string


@as_inp_dataclass_node
@dataclass
class MD:
    """
    Molecular-dynamics (MD) integration and sampling controls for Calphy/LAMMPS runs.

    **Scientific purpose**
    Configure the low-level MD stepping and sampling schedule used by Calphy when computing
    free energies (thermodynamic integration) or running temperature sweeps. These parameters
    control timestep size, inner-loop sampling, repetition, and the strength of thermostat/barostat
    coupling—directly affecting statistical convergence and numerical stability.

    **Typical use-cases**
    * Stabilize MD for solids/liquids prior to free-energy evaluation.
    * Increase sampling (more steps/cycles) to reduce noise in free energies.
    * Tune thermostat/barostat damping for specific potentials or system sizes.

    Attributes
    ----------
    timestep:
        MD timestep (Calphy input: ``timestep``).
    n_small_steps:
        Number of MD steps in the smallest integration block (``n-small-steps``).
    n_every_steps:
        Sampling stride inside a cycle (``n-every-steps``).
    n_repeat_steps:
        Number of repeats per cycle (``n-repeat-steps``).
    n_cycles:
        Number of cycles to perform (``n-cycles``).
    thermostat_damping:
        Thermostat damping parameter (``thermostat-damping``).
    barostat_damping:
        Barostat damping parameter (``barostat-damping``).

    See Also
    --------
    https://calphy.org/en/latest/inputfile.html
    """
    timestep: float = 0.001
    n_small_steps: int = 10000
    n_every_steps: int = 10
    n_repeat_steps: int = 10
    n_cycles: int = 100
    thermostat_damping: float = 0.5
    barostat_damping: float = 0.1

@as_inp_dataclass_node
@dataclass
class NoseHoover:
    """
    Nose–Hoover thermostat/barostat coupling parameters for Calphy.

    **Scientific purpose**
    Set the coupling (damping) timescales for Nose–Hoover temperature and pressure control
    used during equilibration and production MD. These values influence how aggressively the
    system is driven toward the target T/P and can affect equilibration quality and fluctuations.

    **Typical use-cases**
    * NPT equilibration for solids/liquids before free-energy calculations.
    * Adjust damping to improve stability (avoid oscillations) for stiff systems.

    Attributes
    ----------
    thermostat_damping:
        Nose–Hoover thermostat damping (Calphy: ``nose-hoover-thermostat-damping``).
    barostat_damping:
        Nose–Hoover barostat damping (Calphy: ``nose-hoover-barostat-damping``).

    See Also
    --------
    https://calphy.org/en/latest/inputfile.html
    """
    thermostat_damping: float = 0.1
    barostat_damping: float = 0.1

@as_inp_dataclass_node
@dataclass
class Berendsen:
    """
    Berendsen thermostat/barostat coupling parameters for Calphy.

    **Scientific purpose**
    Provide damping parameters for Berendsen temperature/pressure coupling, which can be
    useful for rapid initial equilibration. (Berendsen coupling does not generate the exact
    NPT ensemble, but is often robust for pre-equilibration.)

    **Typical use-cases**
    * Quick pre-equilibration before switching to Nose–Hoover.
    * Troubleshooting unstable Nose–Hoover settings.

    Attributes
    ----------
    thermostat_damping:
        Berendsen thermostat damping (Calphy: ``berendsen-thermostat-damping``).
    barostat_damping:
        Berendsen barostat damping (Calphy: ``berendsen-barostat-damping``).

    See Also
    --------
    https://calphy.org/en/latest/inputfile.html
    """
    thermostat_damping: float = 100.0
    barostat_damping: float = 100.0

@as_inp_dataclass_node
@dataclass
class Tolerance:
    """
    Convergence/tolerance criteria for Calphy solid/liquid free-energy workflows.

    **Scientific purpose**
    Control acceptance thresholds used internally by Calphy to decide whether the system is
    sufficiently “solid-like” or “liquid-like”, whether pressure is acceptable, and how strong
    the spring coupling should be in reference calculations. These tolerances can determine
    whether a run proceeds, repeats, or flags issues.

    **Typical use-cases**
    * Make solid/liquid identification stricter or more permissive.
    * Tune spring constant tolerance for Einstein-crystal-like reference steps.
    * Stabilize workflows when borderline melting/solidification occurs.

    Attributes
    ----------
    spring_constant:
        Spring constant used/checked for tolerance (Calphy: ``tol-spring-constant``).
    solid_fraction:
        Minimum fraction classified as solid (Calphy: ``tol-solid-fraction``).
    liquid_fraction:
        Maximum fraction classified as liquid (Calphy: ``tol-liquid-fraction``).
    pressure:
        Acceptable pressure tolerance (Calphy: ``tol-pressure``).

    See Also
    --------
    https://calphy.org/en/latest/inputfile.html
    """
    spring_constant: float = 0.01
    solid_fraction: float = 0.7
    liquid_fraction: float = 0.05
    pressure: float = 1.0

@as_inp_dataclass_node
@dataclass
class InputClass:
    """
    High-level Calphy calculation input bundle (thermodynamic integration / temperature sweep).

    **Scientific purpose**
    Collect all key parameters needed to run Calphy free-energy calculations for solids and
    liquids with an interatomic potential and an ASE `Atoms` structure. This input object
    is converted into a `calphy.input.Calculation` and used to generate simulation folders,
    run LAMMPS, and postprocess free energies.

    **What this controls**
    - Target state point(s): temperature, pressure, and optional temperature sweep range.
    - Ensemble and equilibration: NPT toggle, equilibration/switching/print steps, iterations.
    - Thermostat/barostat model: Nose–Hoover vs Berendsen selection and parameters.
    - MD stepping and tolerances via nested dataclasses (MD, Tolerance, ...).
    - Parallel resources via ``cores`` (mapped to Calphy ``queue.cores``).

    **Typical use-cases**
    * Compute absolute free energy of a solid at (T, P).
    * Compute absolute free energy of a liquid at (T, P).
    * Run a temperature sweep to obtain f(T) curves for melting point estimation.

    Attributes
    ----------
    md, tolerance, nose_hoover, berendsen:
        Optional nested parameter blocks; if left as ``None`` defaults are filled in by the
        workflow before running Calphy.
    pressure:
        Target pressure (Calphy: ``pressure``).
    temperature:
        Start/target temperature in K (Calphy: ``temperature``).
    temperature_stop:
        End temperature in K for temperature sweeps (used when mode='ts').
    npt:
        Whether to run in NPT (Calphy: ``npt``).
    n_equilibration_steps, n_switching_steps, n_print_steps, n_iterations:
        Core Calphy run-length controls.
    equilibration_control:
        Which equilibration controller to use (e.g. ``"nose-hoover"``).
    melting_cycle:
        Enable/disable melting-cycle logic in Calphy.
    cores:
        Number of CPU cores to request/use (mapped to Calphy queue settings).

    See Also
    --------
    https://calphy.org/en/latest/inputfile.html
    """
    md: Optional[MD] = None 
    tolerance: Optional[Tolerance] = None
    nose_hoover: Optional[NoseHoover] = None
    berendsen: Optional[Berendsen] = None
    pressure: int = 0
    temperature: int = 300
    temperature_stop: int = 600
    npt: bool = True
    n_equilibration_steps: int = 2500
    n_switching_steps: int = 2500
    n_print_steps: int = 1000
    n_iterations: int = 1
    equilibration_control: str = "nose-hoover"
    melting_cycle: bool = False
    cores: Optional[int] = 1

def _generate_random_string(length: int) -> str:
    """Generate a random string of uppercase letters and digits.

    Args:
        length (int): Length of the random string.

    Returns:
        str: Random string of specified length.
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def _prepare_potential_and_structure(
    potential, 
    structure : Atoms
):
    """Prepare the potential and structure for calphy calculations.
    
    Args:
        potential (str): Potential name or DataFrame.
        structure (Atoms): Atomic structure.

    Returns:
        pair_style, pair_coeff, elements, masses, file_name
    """

    import os
    import shutil
    from ase.data import atomic_masses, atomic_numbers
    from pyiron_lammps.potential import get_potential_by_name
    from pyiron_lammps.structure import (
        LammpsStructure,
    ) 
    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    if isinstance(potential, str):
        potential = get_potential_by_name(potential_name=potential)

    elif isinstance(potential, OutputEngine):
        potential = potential.calculator()
        potential = get_potential_by_name(potential_name=potential)

    else: print('TYPE: ', type(potential))

    pair_style = []
    pair_coeff = []
    
    pair_style.append(" ".join(potential["Config"][0].strip().split()[1:]))
    pair_coeff.append(" ".join(potential["Config"][1].strip().split()[1:]))

    #now prepare the list of elements
    elements = list(potential["Species"])
    elements_from_pot = list(potential["Species"])

    lmp_structure = LammpsStructure()
    lmp_structure.potential = potential
    lmp_structure.atom_type = "atomic"
    lmp_structure.el_eam_lst = list(potential["Species"])
    lmp_structure.structure = structure

    # elements_object_lst = structure.get_species_objects()
    elements_struct_lst = structure.get_chemical_symbols()

    masses = []
    for element_name in elements_from_pot:
        if element_name in elements_struct_lst:
            index = list(elements_struct_lst).index(element_name)
            masses.append(atomic_masses[atomic_numbers[element_name]])
        else:
            masses.append(1.0)

    file_name = os.path.join(os.getcwd(), _generate_random_string(7)+'.dat')
    lmp_structure.write_file(file_name=file_name)

    return pair_style, pair_coeff, elements, masses, file_name

def _prepare_input(
    input_class, 
    potential, 
    structure: Atoms, 
    mode='fe', 
    reference_phase='solid'
) -> Calculation:
    """Prepare the input for calphy calculations.

    Args:
    input_class (InputClass): Input parameters for calphy calculations.
    potential (str): Potential name or DataFrame.
    structure (Atoms): Atomic structure.
    mode (str): Calculation mode, either 'fe' for free energy or 'ts' for temperature sweep.
    reference_phase (str): Reference phase, either 'solid' or 'liquid'.

    Returns:
        Calculation: Calphy Calculation object containing the input parameters for the calculation.
    """

    from calphy.input import Calculation

    (
        pair_style, 
        pair_coeff, 
        elements, 
        masses, 
        file_name
     ) = _prepare_potential_and_structure(
        potential=potential, 
        structure=structure
        )

    inpdict = asdict(input_class)
    inpdict["pair_style"] = pair_style
    inpdict["pair_coeff"] = pair_coeff
    inpdict["element"] = elements
    inpdict["mass"] = masses
    inpdict['mode'] = mode
    inpdict['reference_phase'] = reference_phase
    inpdict['lattice'] = file_name
    inpdict["queue"] = {"cores": inpdict["cores"],}
    del inpdict["cores"]

    if inpdict["md"] is None:
        inpdict["md"] = {
                "timestep": 0.001,
                "n_small_steps": 10000,
                "n_every_steps": 10,
                "n_repeat_steps": 10,
                "n_cycles": 100,
                "thermostat_damping": 0.5,
                "barostat_damping": 0.1,
        }
    if inpdict["tolerance"] is None:
        inpdict["tolerance"] = {
                "spring_constant": 0.01,
                "solid_fraction": 0.7,
                "liquid_fraction": 0.05,
                "pressure": 1.0,
        }
    if inpdict["nose_hoover"] is None:
        inpdict["nose_hoover"] = {
                "thermostat_damping": 0.1,
                "barostat_damping": 0.1,
        }
    if inpdict["berendsen"] is None:
        inpdict["berendsen"] = {
                "thermostat_damping": 100.0,
                "barostat_damping": 100.0,
        }
    if mode == 'ts':
        inpdict["temperature"] = [inpdict['temperature'], inpdict["temperature_stop"]]
        del inpdict["temperature_stop"]

    calc = Calculation(**inpdict)
    return calc

def _run_cleanup(simfolder: str, lattice: str, delete_folder: bool = False):
    """Clean up the simulation folder and lattice file.
    
    Args:
        simfolder (str): Simulation folder path.
        lattice (str): Lattice file path.
        delete_folder (bool): Whether to delete the simulation folder.
    """
    import shutil
    import os

    os.remove(path=lattice)

    if delete_folder:
        shutil.rmtree(path=simfolder)

@as_function_node
def SolidFreeEnergy(
    input_class, 
    structure: Atoms, 
    potential,
    delete_folder: bool = False,
    store: bool = True
) -> float:
    """
    Compute the absolute free energy of a **solid** using Calphy thermodynamic integration.

    **Scientific purpose**
    Obtain the Gibbs/Helmholtz free energy (as reported by Calphy) for a crystalline solid
    at a target temperature and pressure, using an interatomic potential and an ASE `Atoms`
    structure. This is commonly used to compare solid polymorph stability and to determine
    melting points via solid–liquid free-energy crossings.

    **Required inputs**
    - ``input_class``: Calphy run parameters (T, P, MD controls, tolerances, etc.).
    - ``structure``: solid atomic configuration (ASE ``Atoms``).
    - ``potential``: LAMMPS potential identifier (name) or an engine object that can be
      resolved to a potential.

    **Typical use-cases**
    * Solid free energy at (T, P) for phase stability ranking.
    * Solid vs liquid free-energy comparison for melting temperature estimation.
    * Generate f(T) datasets by repeating at multiple temperatures.

    Returns
    -------
    float
        Free energy in eV/atom (as reported by Calphy).

    Notes
    -----
    Creates a temporary LAMMPS structure file and a simulation folder. If ``delete_folder``
    is True, the simulation folder is removed after completion.
    """
    from calphy.solid import Solid
    from calphy.routines import routine_fe

    calc = _prepare_input(
        input_class=input_class, 
        potential=potential, 
        structure=structure, 
        mode='fe', 
        reference_phase='solid'
    )

    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job=job)
    _run_cleanup(
        simfolder=simfolder, 
        lattice=calc.lattice, 
        delete_folder=delete_folder
        )
    free_energy = job.report["results"]["free_energy"].tolist()
    return free_energy

@as_function_node
def LiquidFreeEnergy(
    input_class, 
    structure: Atoms, 
    potential, 
    delete_folder: bool = False,
    store: bool = True
) -> float:
    """
    Compute the absolute free energy of a **liquid** using Calphy thermodynamic integration.

    **Scientific purpose**
    Obtain the free energy of a liquid reference state at a target temperature and pressure.
    Together with the corresponding solid free energy, this enables melting point estimation
    by identifying the temperature where f_solid(T) = f_liquid(T).

    **Required inputs**
    - ``input_class``: Calphy run parameters (T, P, MD controls, tolerances, etc.).
    - ``structure``: liquid atomic configuration (ASE ``Atoms``), typically pre-melted/relaxed.
    - ``potential``: LAMMPS potential identifier (name) or resolvable potential object.

    **Typical use-cases**
    * Liquid free energy at (T, P) for solid–liquid equilibrium.
    * Build liquid f(T) curves for melting point / coexistence analysis.

    Returns
    -------
    float
        Free energy in eV/atom (as reported by Calphy).

    Notes
    -----
    Creates a temporary LAMMPS structure file and a simulation folder. If ``delete_folder``
    is True, the simulation folder is removed after completion.
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_fe
    
    calc = _prepare_input(
        input_class=input_class, 
        potential=potential,
        structure=structure, 
        mode='fe', 
        reference_phase='liquid'
        )
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job=job)

    _run_cleanup(
        simfolder=simfolder, 
        lattice=calc.lattice, 
        delete_folder=delete_folder
        )
    free_energy = job.report["results"]["free_energy"].tolist()
    return free_energy

@as_function_node
def SolidFreeEnergyWithTemp(
    input_class, 
    structure: Atoms, 
    potential, 
    delete_folder: bool = False,
    store: bool = True
):
    """
    Run a **temperature sweep** in Calphy and return solid free energy as f(T).

    **Scientific purpose**
    Generate a free-energy curve for a solid across a temperature interval using Calphy’s
    temperature-sweep mode. This is useful for locating phase transitions (e.g., melting)
    by comparing with a corresponding liquid f(T) curve.

    **Required inputs**
    - ``input_class``: must define ``temperature`` (start) and ``temperature_stop`` (end).
    - ``structure``: solid ASE ``Atoms`` configuration.
    - ``potential``: LAMMPS potential identifier or resolvable potential object.

    **Typical use-cases**
    * Produce solid f(T) for melting point estimation via free-energy crossing.
    * Quickly assess temperature dependence without launching many single-T jobs.

    Returns
    -------
    (list[float], list[float])
        ``temperature`` in K and ``free_energy`` in eV/atom, read from Calphy’s
        ``temperature_sweep.dat`` output.
    """
    from calphy.solid import Solid
    from calphy.routines import routine_ts
    import os
    
    calc = _prepare_input(
        input_class=input_class, 
        potential=potential, 
        structure=structure, 
        mode='ts', 
        reference_phase='solid'
    )
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job=job)

    datafile = os.path.join(os.getcwd(), simfolder, 'temperature_sweep.dat')
    temperature_array, free_energy_array = np.loadtxt(datafile, unpack=True, usecols=(0,1))
    temperature = temperature_array.tolist()
    free_energy = free_energy_array.tolist()

    _run_cleanup(
        simfolder=simfolder,
        lattice=calc.lattice, 
        delete_folder=delete_folder
        )
    
    return temperature, free_energy

@as_function_node
def LiquidFreeEnergyWithTemp(
    input_class, 
    structure: Atoms, 
    potential, 
    delete_folder: bool = False,
    store: bool = True
):
    """
    Run a **temperature sweep** in Calphy and return liquid free energy as f(T).

    **Scientific purpose**
    Generate a free-energy curve for a liquid across a temperature interval using Calphy’s
    temperature-sweep mode. Combined with a solid f(T) curve, this enables determination
    of melting/transition temperatures from free-energy intersections.

    **Required inputs**
    - ``input_class``: must define ``temperature`` (start) and ``temperature_stop`` (end).
    - ``structure``: liquid ASE ``Atoms`` configuration (typically equilibrated liquid).
    - ``potential``: LAMMPS potential identifier or resolvable potential object.

    **Typical use-cases**
    * Produce liquid f(T) for melting point estimation via free-energy crossing.

    Returns
    -------
    (list[float], list[float])
        ``temperature`` in K and ``free_energy`` in eV/atom, read from Calphy’s
        ``temperature_sweep.dat`` output.
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_ts
    import os
    
    calc = _prepare_input(
        input_class=input_class, 
        potential=potential, 
        structure=structure, 
        mode='ts', 
        reference_phase='liquid'
    )
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job=job)
    
    datafile = os.path.join(os.getcwd(), simfolder, 'temperature_sweep.dat')
    temperature_array, free_energy_array = np.loadtxt(datafile, unpack=True, usecols=(0,1))
    temperature = temperature_array.tolist()
    free_energy = free_energy_array.tolist()

    _run_cleanup(
        simfolder=simfolder, 
        lattice=calc.lattice, 
        delete_folder=delete_folder
        )
    return temperature, free_energy

@as_function_node("fig")
def PlotFreeEnergy(temperature: np.ndarray, free_energy: np.ndarray) -> plt.Figure:
    """
    Plot a free-energy curve f(T) from temperature-sweep or multi-point calculations.

    **Scientific purpose**
    Visualize the temperature dependence of free energy to inspect trends, noise, and
    potential crossings between phases (e.g., solid vs liquid).

    **Required inputs**
    - ``temperature``: temperatures in K.
    - ``free_energy``: free energies in eV/atom, same length as ``temperature``.

    **Typical use-cases**
    * Plot solid and liquid f(T) curves prior to estimating melting temperature.
    * Quality control: identify outliers or non-smooth behavior in computed free energies.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the f(T) line plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(temperature, free_energy, label='free energy')
    ax.set_ylabel('Free energy (eV/atom)')
    ax.set_xlabel('Temperature (K)')
    plt.legend(frameon=False)
    
    return fig

@as_function_node
def CalcPhaseTransformationTemp(
    temp_A: np.ndarray, 
    fe_A: np.ndarray, 
    temp_B: np.ndarray,
    fe_B: np.ndarray, 
    fit_order: int = 4
) -> plt.Figure:
    """
    Estimate a phase transformation (crossing) temperature from two free-energy curves.

    **Scientific purpose**
    Given free energies f_A(T) and f_B(T) for two phases (often solid vs liquid),
    fit polynomials and find the temperature where the fitted curves are closest/
    intersect. This is a common practical method to estimate melting temperature
    or polymorphic transition temperature from noisy free-energy data.

    **Required inputs**
    - ``temp_A``, ``fe_A``: temperature (K) and free energy (eV/atom) for phase A.
    - ``temp_B``, ``fe_B``: temperature (K) and free energy (eV/atom) for phase B.
    - ``fit_order``: polynomial order used to smooth data before locating the crossing.

    **Typical use-cases**
    * Estimate melting temperature from Calphy solid/liquid f(T) data.
    * Compare two solid polymorphs and estimate transition temperature.

    Returns
    -------
    matplotlib.figure.Figure
        Plot showing raw data, polynomial fits, and a vertical line at the estimated
        transition temperature.

    Notes
    -----
    If the temperature ranges do not overlap, the method extrapolates and emits warnings.
    The returned value is visual (in the plot); the temperature is not returned explicitly
    by this function.
    """
    import matplotlib.pyplot as plt
    import warnings

    #do some fitting to determine temps
    t1min = np.min(temp_A)
    t2min = np.min(temp_B)
    t1max = np.max(temp_A)
    t2max = np.max(temp_B)

    tmin = np.min([t1min, t2min])
    tmax = np.max([t1max, t2max])

    #warn about extrapolation
    if not t1min == t2min:
        warnings.warn(f'free energy is being extrapolated!')
    if not t1max == t2max:
        warnings.warn(f'free energy is being extrapolated!')

    #now fit
    f1fit = np.polyfit(temp_A, fe_A, fit_order)
    f2fit = np.polyfit(temp_B, fe_B, fit_order)

    #reevaluate over the new range
    fit_t = np.arange(tmin, tmax+1, 1)
    fit_f1 = np.polyval(f1fit, fit_t)
    fit_f2 = np.polyval(f2fit, fit_t)

    #now evaluate the intersection temp
    arg = np.argsort(np.abs(fit_f1-fit_f2))[0]
    phase_transition_temperature = fit_t[arg]

    #warn if the temperature is shady
    if np.abs(phase_transition_temperature-tmin) < 1E-3:
        warnings.warn('It is likely there is no intersection of free energies')
    elif np.abs(phase_transition_temperature-tmax) < 1E-3:
        warnings.warn('It is likely there is no intersection of free energies')

    #plot
    c1lo = '#ef9a9a'
    c1hi = '#b71c1c'
    c2lo = '#90caf9'
    c2hi = '#0d47a1'

    fig, ax = plt.subplots()
    ax.plot(fit_t, fit_f1, color=c1lo, label=f'phase A fit')
    ax.plot(fit_t, fit_f2, color=c2lo, label=f'phase B fit')
    ax.plot(temp_A, fe_A, color=c1hi, label='phase A', ls='dashed')
    ax.plot(temp_B, fe_B, color=c2hi, label='phase B', ls='dashed')
    ax.axvline(phase_transition_temperature, ls='dashed', c='#37474f')
    ax.set_ylabel('Free energy (eV/atom)')
    ax.set_xlabel('Temperature (K)')
    ax.legend(frameon=False)

    return fig

@as_function_node
def CollectResults() -> pd.DataFrame:
    """
    Gather Calphy outputs in the current working directory into a single results table.

    **Scientific purpose**
    Aggregate completed Calphy runs (often many folders/jobs) into one pandas DataFrame for
    downstream analysis: comparing free energies, checking convergence diagnostics, or
    exporting summarized thermodynamic data.

    **Typical use-cases**
    * Collect results after running multiple temperatures or multiple phases.
    * Build a dataset for plotting f(T) curves or fitting transition temperatures.
    * Quickly inspect which jobs completed successfully and what free energies were obtained.

    Returns
    -------
    pandas.DataFrame
        DataFrame produced by ``calphy.postprocessing.gather_results('.')``.
    """
    from calphy.postprocessing import gather_results
    results = gather_results('.')
    return results