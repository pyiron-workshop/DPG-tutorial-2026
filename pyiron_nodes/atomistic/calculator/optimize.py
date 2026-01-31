from dataclasses import dataclass
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from enum import Enum
import numpy as np
from pyiron_core import as_function_node, Node, as_inp_dataclass_node


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
        mode: str | RelaxMode,
        calculator: Node,
        opt: Node,
        structures: list[Atoms]
) -> list[Atoms]:
    from tqdm.auto import tqdm
    mode = RelaxMode(mode)
    relaxed_structures = []
    for structure in tqdm(structures, desc=f"Relax {mode.value}"):
        relaxed_structures.append(
                Relax(mode, calculator, opt, structure).pull()
        )
    return relaxed_structures
            

@as_function_node
def Relax(mode: str | RelaxMode, calculator: Node, opt: Node, structure: Atoms) -> Atoms:
    from ase.optimize import LBFGS
    from ase.calculators.singlepoint import SinglePointCalculator

    if isinstance(mode, str):
        mode = mode.lower()
    mode = RelaxMode(mode)

    structure = structure.copy()

    # FIXME: meh
    match mode:
        case RelaxMode.VOLUME:
            calculator.inputs.use_symmetry = True
            structure.calc = calculator.pull()
        case RelaxMode.FULL | RelaxLoop.INTERNAL:
            calculator.inputs.use_symmetry = False
            structure.calc = calculator.pull()
        case _:
            assert False

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure, logfile="/dev/null")
    lbfgs.run(fmax=opt.inputs.force_tolerance.value, steps=opt.inputs.max_steps.value)
    calc = structure.calc
    structure.calc = SinglePointCalculator(structure, **{
            'energy': calc.get_potential_energy(),
            'forces': calc.get_forces(),
            'stress': calc.get_stress()
    })
    relaxed_structure = structure
    relaxed_structure.constraints.clear()
    return relaxed_structure