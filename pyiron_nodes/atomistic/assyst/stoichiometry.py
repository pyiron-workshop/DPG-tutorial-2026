from ase.atoms import Atoms
from dataclasses import dataclass
from itertools import product
from collections.abc import Sequence
import pandas as pd
import numpy as np
from core import as_function_node
from pyiron_nodes.atomistic.engine.generic import OutputEngine
from typing import Optional


@dataclass(frozen=True)
class Stoichiometry(Sequence):
    """
    Immutable container representing one or more chemical stoichiometries.

    This class is used to describe *sets of compositions* for atomistic
    structure generation and high-throughput materials screening workflows.
    Each entry in the container is a dictionary mapping chemical element
    symbols to integer atom counts (e.g. ``{"Fe": 2, "O": 3}``).

    The class supports algebra-like operations that are commonly used in
    compositional design spaces:

    - Addition (``+``): concatenation of stoichiometry lists
    - Inner product (``|``): element-wise combination of compatible stoichiometries
    - Outer product (``*``): Cartesian product of compatible stoichiometries

    These operations are especially useful when constructing multi-element
    composition grids for crystal structure sampling, phase space exploration,
    or workflow graph nodes.

    Attributes:
        stoichiometry (tuple[dict[str, int]]):
            Tuple of stoichiometry dictionaries. Each dictionary corresponds
            to one candidate composition.
    """

    stoichiometry: tuple[dict[str, int]]

    @property
    def elements(self) -> set[str]:
        """
        Return the set of all chemical elements present in the stoichiometries.

        Returns:
            set[str]: Unique element symbols appearing in any stoichiometry.
        """
        e = set()
        for s in self.stoichiometry:
            s = e.union(s.keys())
        return s

    def __add__(self, other: "Stoichiometry") -> "Stoichiometry":
        """
        Concatenate two stoichiometry collections.

        This operation extends the list of candidate compositions without
        modifying or combining individual stoichiometries.

        Args:
            other (Stoichiometry): Another stoichiometry collection.

        Returns:
            Stoichiometry: Combined stoichiometry collection.
        """
        return Stoichiometry(self.stoichiometry + other.stoichiometry)

    def __or__(self, other: "Stoichiometry") -> "Stoichiometry":
        """
        Combine stoichiometries element-wise (inner product).

        Each stoichiometry in this object is merged with the corresponding
        stoichiometry in ``other``. This operation requires that both
        collections contain disjoint chemical elements and are aligned
        in length.

        This is typically used to merge independently generated element
        subspaces (e.g. cation and anion stoichiometries).

        Args:
            other (Stoichiometry): Stoichiometries with disjoint elements.

        Returns:
            Stoichiometry: Element-wise merged stoichiometries.

        Raises:
            AssertionError: If stoichiometries share chemical elements.
        """
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only or stoichiometries of different elements!"
        s = ()
        for me, you in zip(self.stoichiometry, other.stoichiometry, strict=False):
            s += (me | you,)
        return Stoichiometry(s)

    def __mul__(self, other: "Stoichiometry") -> "Stoichiometry":
        """
        Combine stoichiometries via Cartesian product (outer product).

        Every stoichiometry in this object is merged with every stoichiometry
        in ``other``. This operation requires that both collections contain
        disjoint chemical elements.

        This is commonly used to construct full composition grids for
        multi-component materials exploration.

        Args:
            other (Stoichiometry): Stoichiometries with disjoint elements.

        Returns:
            Stoichiometry: Cartesian product of merged stoichiometries.

        Raises:
            AssertionError: If stoichiometries share chemical elements.
        """
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only multiply stoichiometries of different elements!"
        s = ()
        for me, you in product(self.stoichiometry, other.stoichiometry):
            s += (me | you,)
        return Stoichiometry(s)

    def __getitem__(self, index: int) -> dict[str, int]:
        """
        Access a single stoichiometry by index.

        Args:
            index (int): Index of the stoichiometry.

        Returns:
            dict[str, int]: Element-to-atom-count mapping.
        """
        return self.stoichiometry[index]

    def __len__(self) -> int:
        """
        Return the number of stoichiometries.

        Returns:
            int: Number of candidate compositions.
        """
        return len(self.stoichiometry)


@as_function_node
def ElementInput(
    element: str,
    min_atoms: int = 1,
    max_atoms: int = 10,
    step_atoms: int = 1,
) -> Stoichiometry:
    """
    Generate a range of stoichiometries for a single chemical element.

    This node defines a one-dimensional compositional search space by
    varying the atom count of a single element. It is typically used
    as a building block for larger composition spaces via stoichiometry
    algebra (e.g. outer products).

    Args:
        element (str): Chemical element symbol (e.g. ``"Fe"``).
        min_atoms (int): Minimum number of atoms.
        max_atoms (int): Maximum number of atoms.
        step_atoms (int): Step size for atom count increments.

    Returns:
        Stoichiometry: Collection of single-element stoichiometries.
    """
    stoichiometry = Stoichiometry(
        tuple({element: i} for i in range(min_atoms, max_atoms + 1, step_atoms))
    )
    return stoichiometry


@as_function_node("df")
def StoichiometryTable(stoichiometry: Stoichiometry) -> pd.DataFrame:
    """
    Convert a stoichiometry collection into a tabular representation.

    Each row corresponds to one composition and each column corresponds
    to a chemical element. Missing elements are represented as NaN.

    This node is useful for inspection, filtering, logging, or interfacing
    with data-driven workflows and machine-learning pipelines.

    Args:
        stoichiometry (Stoichiometry): Input stoichiometry collection.

    Returns:
        pandas.DataFrame: Tabular stoichiometry representation.
    """
    return pd.DataFrame(stoichiometry.stoichiometry)


@as_function_node("filtered")
def FilterSize(
    elements: Stoichiometry,
    min_atoms: Optional[int] = 0,
    max_atoms: Optional[int] = 10,
):
    """
    Filter stoichiometries by total number of atoms.

    This node removes compositions whose total atom count lies outside
    the specified bounds. It is commonly used to restrict structure
    generation to computationally feasible system sizes.

    Args:
        elements (Stoichiometry): Input stoichiometries.
        min_atoms (int, optional): Minimum total atom count.
        max_atoms (int, optional): Maximum total atom count. If ``None``,
            no upper bound is applied.

    Returns:
        Stoichiometry: Filtered stoichiometry collection.
    """
    import math

    if max_atoms is None:
        max_atoms = math.inf
    return Stoichiometry(
        tuple(
            s for s in elements
            if min_atoms <= sum(s.values()) <= max_atoms
        )
    )


def _atoms_to_structures_df(
    structures: list[Atoms],
    spacegroups: list[int],
    calc_type: str = ""
) -> pd.DataFrame:
    """
    Convert ASE Atoms objects into a structured pandas DataFrame.

    This helper function extracts structural, chemical, and energetic
    information from a list of ``ase.Atoms`` objects and stores them in
    a standardized tabular format suitable for storage, post-processing,
    or workflow graph outputs.

    Args:
        structures (list[Atoms]): Atomic structures.
        spacegroups (list[int]): Corresponding space group numbers.
        calc_type (str): Optional calculation type label.

    Returns:
        pandas.DataFrame: DataFrame containing structural and energetic data.
    """
    rows = []

    for i, (atoms, spg) in enumerate(zip(structures, spacegroups)):
        energy = np.nan
        forces = np.nan
        stress = np.nan

        if atoms.calc is not None:
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                stress = atoms.get_stress()
            except Exception:
                pass

        rows.append({
            "name": f"{''.join(set(atoms.get_chemical_symbols()))}{calc_type.capitalize()}_structure_{i}",
            "calc_type": calc_type.capitalize(),
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.positions.copy(),
            "cell": atoms.cell.array.copy(),
            "volume": atoms.get_volume(),
            "pbc": tuple(atoms.pbc),
            "spacegroup": spg,
            "number_of_atoms": len(atoms),
            "energy": energy,
            "forces": forces,
            "stress": stress,
        })

    return pd.DataFrame(rows)


def _structures_df_to_atoms(df: pd.DataFrame) -> list[Atoms]:
    """
    Reconstruct ASE Atoms objects from a structures DataFrame.

    This helper function reverses the transformation performed by
    ``_atoms_to_structures_df`` and is useful for restarting simulations
    or exporting stored structures back into atomistic workflows.

    Args:
        df (pandas.DataFrame): DataFrame containing structure information.

    Returns:
        list[Atoms]: Reconstructed atomic structures.
    """
    structures = []
    for _, row in df.iterrows():
        atoms = Atoms(
            symbols=row["symbols"],
            positions=row["positions"],
            cell=row["cell"],
            pbc=row["pbc"],
        )
        structures.append(atoms)

    return structures


@as_function_node
def SpaceGroupSampling(
    elements: Stoichiometry,
    spacegroups: list[int] | tuple[int, ...] | None = None,
    max_atoms: int = 10,
    max_structures: Optional[int] = 50,
    engine: OutputEngine = None,
    store: bool = False
) -> pd.DataFrame:
    """
    Generate and evaluate crystal structures using space-group sampling.

    This node performs symmetry-aware random structure generation based
    on provided stoichiometries and space groups. Generated structures
    are evaluated using an atomistic calculator and returned in a
    standardized tabular format.

    Typical use cases include:
    - High-throughput crystal structure prediction
    - Symmetry-constrained structure enumeration
    - Initial structure generation for DFT or ML potentials

    Args:
        elements (Stoichiometry): Candidate compositions per structure.
        spacegroups (list[int] or tuple[int], optional): Space group numbers
            to sample. If ``None``, all 230 space groups are considered.
        max_atoms (int): Maximum allowed number of atoms per structure.
        max_structures (int, optional): Maximum number of structures to generate.
        engine (OutputEngine, optional): Atomistic engine providing a calculator.
            If ``None``, a default GRACE engine is used.
        store (bool): Whether to store results in the engine backend.

    Returns:
        pandas.DataFrame: Table of generated structures including energies,
        forces, stresses, and symmetry information.
    """
    from warnings import catch_warnings
    from structuretoolkit.analyse import get_symmetry
    from tqdm.auto import tqdm
    from assyst.crystals import pyxtal
    from ase.calculators.singlepoint import SinglePointCalculator
    import math

    if engine is None:
        from pyiron_nodes.atomistic.engine.grace import GRACE
        print("No Engine is used, loading GRACE engine!")
        engine = GRACE().run()

    calculator = engine.calculator

    if spacegroups is None:
        spacegroups = list(range(1, 231))

    if max_structures == "":
        max_structures = math.inf

    structures = []
    spg_numbers = []

    with catch_warnings(category=UserWarning, action="ignore"):
        for stoich in (bar := tqdm(elements)):
            elements_, num_atomss = zip(*stoich.items())
            stoich_str = "".join(f"{s}{n}" for s, n in zip(elements_, num_atomss))
            bar.set_description(stoich_str)

            for s in pyxtal(spacegroups, elements_, num_atomss):
                atoms = s["atoms"].copy()
                atoms.calc = calculator
                structures.append(atoms)
                spg_numbers.append(get_symmetry(atoms).info["number"])

            if len(structures) > max_structures:
                print("Maximum number of structures reached! Ending the generation..")
                structures = structures[:max_structures]
                spg_numbers = spg_numbers[:max_structures]
                break

    for structure in tqdm(structures, desc="Spacegroup Calculations"):
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
        stress = structure.get_stress()

        structure.calc = SinglePointCalculator(
            structure,
            energy=energy,
            forces=forces,
            stress=stress,
        )

    df_structures = _atoms_to_structures_df(
        structures=structures,
        spacegroups=spg_numbers
    )
    return df_structures