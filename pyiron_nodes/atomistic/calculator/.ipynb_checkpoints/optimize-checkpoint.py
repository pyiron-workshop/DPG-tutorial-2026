from dataclasses import dataclass
from typing import Literal
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from enum import Enum
import numpy as np
import pandas as pd
from pyiron_nodes.atomistic.assyst.stoichiometry import _structures_df_to_atoms, _atoms_to_structures_df
from core import as_function_node, as_inp_dataclass_node
from pyiron_nodes.atomistic.engine.generic import OutputEngine


@as_inp_dataclass_node
@dataclass
class GenericOptimizerSettings:
    max_steps: int = 10
    force_tolerance: float = 1e-2


class RelaxMode(Enum):
    VOLUME = "volume"
    # CELL = "cell"
    INTERNAL = "internal"
    FULL = "full"

    def apply_filter_and_constraints(self, structure):
        match self:
            case RelaxMode.VOLUME:
                structure.set_constraint(FixAtoms(np.ones(len(structure),dtype=bool)))
                return FrechetCellFilter(structure, hydrostatic_strain=True)
            case RelaxMode.INTERNAL:
                return structure
            case RelaxMode.FULL:
                return FrechetCellFilter(structure)
            case _:
                raise ValueError("Lazy Marvin")


@as_function_node
def RelaxLoop(
        mode: Literal['volume', 'full'],
        opt,
        df_structures: pd.DataFrame,
        engine: OutputEngine = None,
        store: bool = False) -> pd.DataFrame:
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
        
    df_relaxed = _atoms_to_structures_df(structures = relaxed_structures, spacegroups = spg_numbers, calc_type = mode.value)
    return df_relaxed
            

@as_function_node
def Relax(mode: Literal['volume', 'full'], 
          opt, 
          structure: Atoms,
          engine: OutputEngine) -> Atoms:
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
            # calculator.inputs.use_symmetry = True
            structure.calc = calculator
        case RelaxMode.FULL | RelaxLoop.INTERNAL:
            # calculator.inputs.use_symmetry = False
            structure.calc = calculator
        case _:
            assert False

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure, logfile="/dev/null")
    lbfgs.run(fmax=opt.force_tolerance, steps=opt.max_steps)
    calc = structure.calc
    structure.calc = SinglePointCalculator(structure, **{
            'energy': calc.get_potential_energy(),
            'forces': calc.get_forces(),
            'stress': calc.get_stress()
    })
    relaxed_structure = structure
    relaxed_structure.constraints.clear()
    return relaxed_structure