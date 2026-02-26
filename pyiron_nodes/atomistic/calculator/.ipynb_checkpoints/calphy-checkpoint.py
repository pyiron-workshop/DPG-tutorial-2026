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
    """Molecular dynamics parameters.

    Attributes:
        timestep (float):
            See https://calphy.org/en/latest/inputfile.html#timestep
        n_small_steps (int):
            See https://calphy.org/en/latest/inputfile.html#n-small-steps
        n_every_steps (int): 
            See https://calphy.org/en/latest/inputfile.html#n-every-steps
        n_repeat_steps (int): 
            See https://calphy.org/en/latest/inputfile.html#n-repeat-steps
        n_cycles (int): 
            See https://calphy.org/en/latest/inputfile.html#n-cycles
        thermostat_damping (float): 
            See https://calphy.org/en/latest/inputfile.html#thermostat-damping
        barostat_damping (float): 
            See https://calphy.org/en/latest/inputfile.html#barostat-damping
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
    """Nose-Hoover parameters.

    Attributes:
        thermostat_damping (float):
            See https://calphy.org/en/latest/inputfile.html#nose-hoover-thermostat-damping
        barostat_damping (float):
            See https://calphy.org/en/latest/inputfile.html#nose-hoover-barostat-damping
    """
    thermostat_damping: float = 0.1
    barostat_damping: float = 0.1

@as_inp_dataclass_node
@dataclass
class Berendsen:
    """Berendsen parameters.

    Attributes:
        thermostat_damping (float):
            See https://calphy.org/en/latest/inputfile.html#berendsen-thermostat-damping
        barostat_damping (float):
            See https://calphy.org/en/latest/inputfile.html#berendsen-barostat-damping
    """
    thermostat_damping: float = 100.0
    barostat_damping: float = 100.0

@as_inp_dataclass_node
@dataclass
class Tolerance:
    """Tolerance parameters.

    Attributes:
        spring_constant (float):
            See https://calphy.org/en/latest/inputfile.html#tol-spring-constant
        solid_fraction (float):
            See https://calphy.org/en/latest/inputfile.html#tol-solid-fraction
        liquid_fraction (float):
            See https://calphy.org/en/latest/inputfile.html#tol-liquid-fraction
        pressure (float):
            See https://calphy.org/en/latest/inputfile.html#tol-pressure
    """
    spring_constant: float = 0.01
    solid_fraction: float = 0.7
    liquid_fraction: float = 0.05
    pressure: float = 1.0

@as_inp_dataclass_node
@dataclass
class InputClass:
    """Input parameters for calphy calculations.

    Attributes:
        md (MD): Molecular dynamics parameters.
        tolerance (Tolerance): Tolerance parameters.
        nose_hoover (NoseHoover): Nose-Hoover parameters.
        berendsen (Berendsen): Berendsen parameters.
        queue (Queue): Queue parameters.
        pressure (int):
            See https://calphy.org/en/latest/inputfile.html#pressure
        temperature (int):
            See https://calphy.org/en/latest/inputfile.html#temperature
        npt (bool):
            See https://calphy.org/en/latest/inputfile.html#npt
        n_equilibration_steps (int):
            See https://calphy.org/en/latest/inputfile.html#n-equilibration-steps
        n_switching_steps (int):
            See https://calphy.org/en/latest/inputfile.html#n-switching-steps
        n_print_steps (int):
            See https://calphy.org/en/latest/inputfile.html#n-print-steps
        n_iterations (int):
            See https://calphy.org/en/latest/inputfile.html#n-iterations
        equilibration_control (str):
            See https://calphy.org/en/latest/inputfile.html#equilibration-control
        melting_cycle (bool):
            See https://calphy.org/en/latest/inputfile.html#melting-cycle
        spring_constants (Optional[float]):
            See https://calphy.org/en/latest/inputfile.html#spring-constants        
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
    potential: str, 
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

    if isinstance(potential, str):
        potential = get_potential_by_name(potential_name=potential)

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
    potential: str, 
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
    potential: str,
    delete_folder: bool = False,
    store: bool = True
) -> float:
    """Calculate the free energy of a solid phase.

    Args:
        input_class (InputClass): Input parameters for calphy calculations.
        structure (Atoms): Atomic structure.
        potential (str): Potential name or DataFrame.
        delete_folder (bool): Whether to delete the simulation folder after calculation.
    
    Returns:
        float: Free energy in eV/atom
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
    potential: str, 
    delete_folder: bool = False,
    store: bool = True
) -> float:
    """Calculate the free energy of a liquid phase.

    Args:
        input_class (InputClass): Input parameters for calphy calculations.
        structure (Atoms): Atomic structure.
        potential (str): Potential name or DataFrame.
        delete_folder (bool): Whether to delete the simulation folder after calculation.
    
    Returns:
        float: Free energy in eV/atom
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
    potential: str, 
    delete_folder: bool = False,
    store: bool = True
):
    """Calculate the free energy of a solid phase as a function of temperature.

    Args:
        input_class (InputClass): Input parameters for calphy calculations.
        structure (Atoms): Atomic structure.
        potential (str): Potential name or DataFrame.
        delete_folder (bool): Whether to delete the simulation folder after calculation.

    Returns:
        Temperature and free energy in K and eV/atom, respectively.
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
    potential: str, 
    delete_folder: bool = False,
    store: bool = True
):
    """Calculate the free energy of a liquid phase as a function of temperature.

    Args:
        input_class (InputClass): Input parameters for calphy calculations.
        structure (Atoms): Atomic structure.
        potential (str): Potential name or DataFrame.
        delete_folder (bool): Whether to delete the simulation folder after calculation.

    Returns:
        Temperature and free energy in K and eV/atom, respectively.
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
    """Plot the free energy as a function of temperature.

    Args:
        temperature (np.ndarray): Temperature array in K.
        free_energy (np.ndarray): Free energy array in eV/atom.
    
    Returns:
        plt.Figure: Figure showing the free energy as a function of temperature.
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
    """Calculate the phase transformation temperature from free energy data.

    Args:
        temp_A (np.ndarray): Temperature array for phase 1.
        fe_A (np.ndarray): Free energy array for phase 1.
        temp_B (np.ndarray): Temperature array for phase 2.
        fe_B (np.ndarray): Free energy array for phase 2.
        fit_order (int): Order of the polynomial fit.
    
    Returns:
        plt.Figure: Figure showing the phase transformation temperature.
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
    """Collect the results from calphy calculations.

    Returns:
        pd.DataFrame: DataFrame containing the results of the calculations.
    """
    from calphy.postprocessing import gather_results
    results = gather_results('.')
    return results