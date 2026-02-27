from dataclasses import dataclass
from typing import Literal
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from enum import Enum
import numpy as np
import pandas as pd
from pyiron_nodes.atomistic.assyst.stoichiometry import (
    _structures_df_to_atoms,
    _atoms_to_structures_df,
)
from core import as_function_node, as_inp_dataclass_node
from pyiron_nodes.atomistic.engine.generic import OutputEngine


@as_inp_dataclass_node
@dataclass
class GenericOptimizerSettings:
    """
    Generic optimizer settings for atomistic structure relaxation.

    This input dataclass defines convergence criteria and iteration
    limits for geometry optimization algorithms used in atomistic
    simulations. It is designed to be passed into relaxation nodes
    as a structured configuration object.

    Attributes:
        max_steps (int):
            Maximum number of optimization steps.
        force_tolerance (float):
            Convergence threshold for maximum force (eV/Å).
    """
    max_steps: int = 10
    force_tolerance: float = 1e-2


class RelaxMode(Enum):
    """
    Enumeration of supported relaxation modes.

    Relaxation modes define which degrees of freedom are optimized
    during structural relaxation, such as atomic positions, cell
    volume, or the full cell shape.

    Members:
        VOLUME:
            Relax only the cell volume while keeping atomic positions fixed.
        INTERNAL:
            Relax atomic positions only (cell fixed).
        FULL:
            Relax both atomic positions and full cell degrees of freedom.
    """
    VOLUME = "volume"
    # CELL = "cell"
    INTERNAL = "internal"
    FULL = "full"

    def apply_filter_and_constraints(self, structure):
        """
        Apply ASE filters and constraints according to the relaxation mode.

        This method prepares the structure for optimization by attaching
        appropriate constraints and cell filters depending on which degrees
        of freedom are to be relaxed.

        Args:
            structure (Atoms):
                ASE atoms object to be prepared for relaxation.

        Returns:
            Atoms or ase.filters.Filter:
                Filtered structure compatible with ASE optimizers.

        Raises:
            ValueError:
                If an unsupported relaxation mode is encountered.
        """
        match self:
            case RelaxMode.VOLUME:
                structure.set_constraint(
                    FixAtoms(np.ones(len(structure), dtype=bool))
                )
                return FrechetCellFilter(structure, hydrostatic_strain=True)
            case RelaxMode.INTERNAL:
                return structure
            case RelaxMode.FULL:
                return FrechetCellFilter(structure)
            case _:
                raise ValueError("Lazy Marvin")


@as_function_node
def RelaxLoop(
    mode: Literal["volume", "full"],
    opt,
    df_structures: pd.DataFrame,
    engine: OutputEngine = None,
    store: bool = False,
) -> pd.DataFrame:
    """
    Relax a dataset of structures using a specified relaxation mode.

    This node iterates over a DataFrame of atomistic structures,
    performs geometry relaxation for each structure, and returns
    the relaxed configurations in a standardized tabular format.

    Typical use cases include:
    - Bulk volume relaxation
    - Full structural relaxation prior to property evaluation
    - Post-processing of generated or perturbed structures

    Args:
        mode ({"volume", "full"}):
            Relaxation mode specifying which degrees of freedom are optimized.
        opt:
            Optimizer settings object (e.g. ``GenericOptimizerSettings``).
        df_structures (pandas.DataFrame):
            Input DataFrame containing structures to be relaxed.
        engine (OutputEngine, optional):
            Atomistic engine providing an ASE calculator. If ``None``,
            a default GRACE engine is used.
        store (bool):
            Whether to store outputs in a workflow backend.

    Returns:
        pandas.DataFrame:
            DataFrame containing relaxed structures with updated
            energies, forces, stresses, and symmetry information.
    """
    from tqdm.auto import tqdm
    from structuretoolkit.analyse import get_symmetry

    if engine is None:
        from pyiron_nodes.atomistic.engine.grace import GRACE
        print("No Engine is used, loading GRACE engine!")
        engine = GRACE().run()

    mode = RelaxMode(mode)

    structures = _structures_df_to_atoms(df_structures)
    relaxed_structures = []
    spg_numbers = []

    for structure in tqdm(structures, desc=f"Relax {mode.value}"):
        relaxed_st = Relax(mode, opt, structure, engine).run()
        relaxed_structures.append(relaxed_st)
        spg_numbers.append(get_symmetry(relaxed_st).info["number"])

    df_relaxed = _atoms_to_structures_df(
        structures=relaxed_structures,
        spacegroups=spg_numbers,
        calc_type=mode.value,
    )
    return df_relaxed


@as_function_node
def Relax(
    mode: Literal["volume", "full"],
    opt,
    structure: Atoms,
    engine: OutputEngine,
) -> Atoms:
    """
    Relax a single atomic structure using an ASE optimizer.

    This node performs geometry optimization on an individual structure
    according to the selected relaxation mode. It supports constrained
    volume relaxation and full cell relaxation using ASE's LBFGS optimizer.

    The relaxed structure is returned with energies, forces, and stresses
    frozen via a single-point calculator.

    Args:
        mode ({"volume", "full"}):
            Relaxation mode controlling which degrees of freedom are optimized.
        opt:
            Optimizer settings object providing convergence criteria.
        structure (Atoms):
            Input ASE atoms object to be relaxed.
        engine (OutputEngine):
            Atomistic engine providing an ASE calculator.

    Returns:
        Atoms:
            Relaxed atomic structure with single-point results attached.
    """
    from ase.optimize import LBFGS
    from ase.calculators.singlepoint import SinglePointCalculator

    if isinstance(mode, str):
        mode = mode.lower()
    mode = RelaxMode(mode)

    structure = structure.copy()
    calculator = engine.calculator

    # FIXME: meh
    match mode:
        case RelaxMode.VOLUME:
            structure.calc = calculator
        case RelaxMode.FULL | RelaxLoop.INTERNAL:
            structure.calc = calculator
        case _:
            assert False

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure, logfile="/dev/null")
    lbfgs.run(fmax=opt.force_tolerance, steps=opt.max_steps)

    calc = structure.calc
    structure.calc = SinglePointCalculator(
        structure,
        **{
            "energy": calc.get_potential_energy(),
            "forces": calc.get_forces(),
            "stress": calc.get_stress(),
        },
    )

    relaxed_structure = structure
    relaxed_structure.constraints.clear()
    return relaxed_structure

@as_function_node
def SinglePointStatic(structure:Atoms,
                 engine: OutputEngine,
                opt:dict | None = None):
    
    structure_temp = structure.copy()
    
    structure_temp.calc = engine.calculator
    energy = structure_temp.get_potential_energy()
    volume = structure_temp.get_volume()

    return energy, volume, structure_temp