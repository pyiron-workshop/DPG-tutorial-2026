from ase.atoms import Atoms
from enum import Enum
import numpy as np
import pandas as pd
from core import as_function_node
from pyiron_nodes.atomistic.assyst.stoichiometry import _structures_df_to_atoms, _atoms_to_structures_df
from pyiron_nodes.atomistic.engine.generic import OutputEngine




def rattle(structure: Atoms, sigma: float) -> Atoms:
    """Randomly displace positions with gaussian noise.

    Operates INPLACE."""
    structure.rattle(stdev=sigma)
    return structure


def stretch(structure: Atoms, hydro: float, shear: float) -> Atoms:
    """Randomly stretch cell with uniform noise.

    Operates INPLACE."""
    strain = shear * (2 * np.random.rand(3, 3) - 1)
    strain = 0.5 * (strain + strain.T)  # symmetrize
    np.fill_diagonal(strain, 1 + hydro * (2 * np.random.rand(3) - 1))
    structure.set_cell(structure.cell.array @ strain, scale_atoms=True)
    return structure


@as_function_node
def RattleAndStrech(structure: Atoms, sigma: float, samples: int) -> list[Atoms]:
    structures = []
    # no point in rattling single atoms
    if len(structure) > 1:
        for _ in range(samples):
            structures.append(
                    stretch(
                        rattle(structure.copy(), sigma),
                        hydro=0.05, shear=0.005
                    )
            )
    return structures


@as_function_node
def Rattle(structure, seed: int = 42, stdev: float = 0.1):
    """
    Randomly displace the atoms in the structure.

    Parameters:

    seed = 42:
    Random seed for the random number generator.
    amplitude = 0.1:
    The amplitude of the random displacement.
    """
    structure = structure.copy()
    structure.rattle(seed=seed, stdev=stdev)
    return structure


@as_function_node
def RattleLoop(
        df_structures: pd.DataFrame, sigma: float = 0.25, samples: int = 4,
        engine: OutputEngine = None, 
        store: bool = False
) -> pd.DataFrame:
    """
    
    """
    from structuretoolkit.analyse import get_symmetry
    from tqdm.auto import tqdm
    from ase.calculators.singlepoint import SinglePointCalculator
    
    if engine is None:
        from pyiron_nodes.atomistic.engine.grace import GRACE
        print("No Engine is used, loading GRACE engine!")
        engine = GRACE().run()
    calculator = engine.calculator
    structures = _structures_df_to_atoms(df_structures)
    rattled_structures = []
    spg_numbers = []
    for structure in structures:
        structure = structure.copy()
        rattled_st = RattleAndStrech(structure, sigma, samples).run()
        rattled_structures += rattled_st
        for single_rattle in rattled_st:
            spg_numbers.append(get_symmetry(single_rattle).info["number"])
            single_rattle.calc = calculator

    for structure in tqdm(rattled_structures, desc="Rattle Calculations"):
        # Force evaluation THROUGH Atoms
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
        stress = structure.get_stress()
    
        # Freeze results
        structure.calc = SinglePointCalculator(
            structure,
            energy=energy,
            forces=forces,
            stress=stress,
        )
    df_rattled = _atoms_to_structures_df(structures = rattled_structures, spacegroups = spg_numbers, calc_type = "rattle")
    #lazy filter for now
    df_rattled = df_rattled[(df_rattled['energy'] < 20) & (df_rattled['energy'] > -20)]
    return df_rattled

@as_function_node
def Stretch(
        structure: Atoms, hydro: float, shear: float, samples: int,
        hydro_shear_ratio: float = 0.7
) -> list[Atoms]:
    structures = []
    for _ in range(samples):
        if np.random.rand() < hydro_shear_ratio:
            ihydro, ishear = hydro, 0.05
        else:
            ihydro, ishear = 0.05, shear
        structures.append(
                stretch(structure.copy(), hydro=ihydro, shear=ishear)
        )
    return structures


@as_function_node
def StretchLoop(
        df_structures: pd.DataFrame, 
        hydro: float = 0.8, shear: float = 0.2, samples: int = 4,
        hydro_shear_ratio: float = 0.7,
        engine: OutputEngine = None,
        store: bool = False
) -> pd.DataFrame:
    from structuretoolkit.analyse import get_symmetry
    from tqdm.auto import tqdm
    from ase.calculators.singlepoint import SinglePointCalculator
    
    if engine is None:
        from pyiron_nodes.atomistic.engine.grace import GRACE
        print("No Engine is used, loading GRACE engine!")
        engine = GRACE().run()
    calculator = engine.calculator
    structures = _structures_df_to_atoms(df_structures)
    stretched_structures = []
    spg_numbers = []
    for structure in structures:
        structure = structure.copy()
        stretched_st = Stretch(structure, hydro, shear, samples, hydro_shear_ratio).run()
        stretched_structures += stretched_st
        for single_stretch in stretched_st:
            single_stretch.calc = calculator
            spg_numbers.append(get_symmetry(single_stretch).info["number"])
    
    for structure in tqdm(stretched_structures, desc="Stretch Calculations"):
        # Force evaluation THROUGH Atoms
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
        stress = structure.get_stress()
    
        # Freeze results
        structure.calc = SinglePointCalculator(
            structure,
            energy=energy,
            forces=forces,
            stress=stress,
        )
    df_stretched = _atoms_to_structures_df(structures = stretched_structures, spacegroups = spg_numbers, calc_type="stretch")
    #lazy filter for now
    df_stretched = df_stretched[(df_stretched['energy'] < 20) & (df_stretched['energy'] > -20)]
    return df_stretched


@as_function_node("structure")
def Repeat(structure: Atoms, repeat_scalar: int = 1) -> Atoms:
    """
    Repeat a crystal structure periodically along all lattice vectors.

    Parameters
    ----------
    structure : Atoms
        The ASE ``Atoms`` object to be repeated.
    repeat_scalar : int, optional
        Number of repetitions along each lattice vector (default is ``1`` – no change).

    Returns
    -------
    Atoms
        A new ``Atoms`` object containing the repeated supercell.

    Task hint
    ----------
    Use this node when the workflow requires building a larger supercell
    from a primitive cell (e.g., “create a 2×2×2 bulk Al supercell”).
    """
    return structure.repeat(int(repeat_scalar))
