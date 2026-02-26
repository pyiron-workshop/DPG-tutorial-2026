from ase.atoms import Atoms
from enum import Enum
import numpy as np
import pandas as pd
from core import as_function_node
from pyiron_nodes.atomistic.assyst.stoichiometry import (
    _structures_df_to_atoms,
    _atoms_to_structures_df,
)
from pyiron_nodes.atomistic.engine.generic import OutputEngine


def rattle(structure: Atoms, sigma: float) -> Atoms:
    """
    Randomly displace atomic positions using Gaussian noise.

    This helper function applies an in-place positional perturbation
    to an ASE ``Atoms`` object. It is typically used to generate
    distorted structures for data augmentation or sampling around
    local minima.

    Args:
        structure (Atoms):
            ASE atoms object to be rattled.
        sigma (float):
            Standard deviation of the Gaussian displacement in angstrom.

    Returns:
        Atoms:
            The same ``Atoms`` object with perturbed atomic positions.
    """
    structure.rattle(stdev=sigma)
    return structure


def stretch(structure: Atoms, hydro: float, shear: float) -> Atoms:
    """
    Apply a random homogeneous strain to a crystal structure.

    This helper function perturbs the simulation cell using a combination
    of hydrostatic and shear strain and rescales atomic positions
    accordingly. The operation is performed in place.

    Args:
        structure (Atoms):
            ASE atoms object to be strained.
        hydro (float):
            Amplitude of random hydrostatic strain.
        shear (float):
            Amplitude of random shear strain.

    Returns:
        Atoms:
            The same ``Atoms`` object with a strained cell.
    """
    strain = shear * (2 * np.random.rand(3, 3) - 1)
    strain = 0.5 * (strain + strain.T)  # symmetrize
    np.fill_diagonal(strain, 1 + hydro * (2 * np.random.rand(3) - 1))
    structure.set_cell(structure.cell.array @ strain, scale_atoms=True)
    return structure


@as_function_node
def RattleAndStrech(structure: Atoms, sigma: float, samples: int) -> list[Atoms]:
    """
    Generate multiple rattled and strained variants of a structure.

    This node creates a set of perturbed structures by applying random
    atomic displacements (rattling) followed by small random cell
    distortions (stretching). It is typically used for data augmentation
    in training datasets for interatomic potentials.

    Single-atom structures are ignored, as rattling is not meaningful
    in that case.

    Args:
        structure (Atoms):
            Input ASE atoms object.
        sigma (float):
            Standard deviation of atomic displacements.
        samples (int):
            Number of perturbed structures to generate.

    Returns:
        list[Atoms]:
            List of rattled and stretched structures.
    """
    structures = []
    # no point in rattling single atoms
    if len(structure) > 1:
        for _ in range(samples):
            structures.append(
                stretch(
                    rattle(structure.copy(), sigma),
                    hydro=0.05,
                    shear=0.005,
                )
            )
    return structures


@as_function_node
def Rattle(structure, seed: int = 42, stdev: float = 0.1):
    """
    Randomly displace atoms in a structure with a fixed random seed.

    This node creates a rattled copy of an ASE ``Atoms`` object by
    applying Gaussian noise to atomic positions. The random seed
    ensures reproducibility across workflow runs.

    Args:
        structure (Atoms):
            Input atomic structure.
        seed (int):
            Random seed for reproducible displacements.
        stdev (float):
            Standard deviation of the displacement in angstrom.

    Returns:
        Atoms:
            New rattled structure.
    """
    structure = structure.copy()
    structure.rattle(seed=seed, stdev=stdev)
    return structure


@as_function_node
def RattleLoop(
    df_structures: pd.DataFrame,
    sigma: float = 0.25,
    samples: int = 4,
    engine: OutputEngine = None,
    store: bool = False,
) -> pd.DataFrame:
    """
    Apply rattling and stretching to a dataset of structures and evaluate them.

    This node takes a DataFrame of structures, generates multiple rattled
    variants per structure, evaluates energies, forces, and stresses using
    an atomistic engine, and returns the results as a new DataFrame.

    It is commonly used to enrich training datasets with distorted
    configurations for improved potential robustness.

    Args:
        df_structures (pandas.DataFrame):
            Input structure dataset in tabular form.
        sigma (float):
            Standard deviation of atomic displacements.
        samples (int):
            Number of rattled structures generated per input structure.
        engine (OutputEngine, optional):
            Atomistic engine providing an ASE calculator. If ``None``,
            a default GRACE engine is used.
        store (bool):
            Whether to store outputs in a workflow backend.

    Returns:
        pandas.DataFrame:
            DataFrame containing rattled structures with evaluated
            energies, forces, stresses, and space group information.
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
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
        stress = structure.get_stress()

        structure.calc = SinglePointCalculator(
            structure,
            energy=energy,
            forces=forces,
            stress=stress,
        )

    df_rattled = _atoms_to_structures_df(
        structures=rattled_structures,
        spacegroups=spg_numbers,
        calc_type="rattle",
    )

    # lazy filter for now
    df_rattled = df_rattled[
        (df_rattled["energy"] < 20) & (df_rattled["energy"] > -20)
    ]
    return df_rattled


@as_function_node
def Stretch(
    structure: Atoms,
    hydro: float,
    shear: float,
    samples: int,
    hydro_shear_ratio: float = 0.7,
) -> list[Atoms]:
    """
    Generate strained variants of a structure.

    This node creates multiple copies of a structure subjected to
    random hydrostatic or shear strain. The type of strain applied
    to each sample is chosen probabilistically.

    Args:
        structure (Atoms):
            Input ASE atoms object.
        hydro (float):
            Hydrostatic strain amplitude.
        shear (float):
            Shear strain amplitude.
        samples (int):
            Number of strained structures to generate.
        hydro_shear_ratio (float):
            Probability of applying hydrostatic strain
            instead of shear strain.

    Returns:
        list[Atoms]:
            List of strained structures.
    """
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
    hydro: float = 0.8,
    shear: float = 0.2,
    samples: int = 4,
    hydro_shear_ratio: float = 0.7,
    engine: OutputEngine = None,
    store: bool = False,
) -> pd.DataFrame:
    """
    Apply random strain perturbations to a dataset of structures.

    This node generates strained variants for each structure in the
    input DataFrame, evaluates them using an atomistic engine, and
    returns the results in a standardized tabular format.

    It is typically used for stress/strain sampling or augmenting
    training data for interatomic potential fitting.

    Args:
        df_structures (pandas.DataFrame):
            Input structure dataset.
        hydro (float):
            Hydrostatic strain amplitude.
        shear (float):
            Shear strain amplitude.
        samples (int):
            Number of strained structures per input structure.
        hydro_shear_ratio (float):
            Probability of applying hydrostatic vs. shear strain.
        engine (OutputEngine, optional):
            Atomistic engine providing an ASE calculator.
        store (bool):
            Whether to store outputs in a workflow backend.

    Returns:
        pandas.DataFrame:
            DataFrame containing strained structures with evaluated
            energies, forces, stresses, and symmetry information.
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
    stretched_structures = []
    spg_numbers = []

    for structure in structures:
        structure = structure.copy()
        stretched_st = Stretch(
            structure, hydro, shear, samples, hydro_shear_ratio
        ).run()
        stretched_structures += stretched_st
        for single_stretch in stretched_st:
            single_stretch.calc = calculator
            spg_numbers.append(get_symmetry(single_stretch).info["number"])

    for structure in tqdm(stretched_structures, desc="Stretch Calculations"):
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
        stress = structure.get_stress()

        structure.calc = SinglePointCalculator(
            structure,
            energy=energy,
            forces=forces,
            stress=stress,
        )

    df_stretched = _atoms_to_structures_df(
        structures=stretched_structures,
        spacegroups=spg_numbers,
        calc_type="stretch",
    )

    # lazy filter for now
    df_stretched = df_stretched[
        (df_stretched["energy"] < 20) & (df_stretched["energy"] > -20)
    ]
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
