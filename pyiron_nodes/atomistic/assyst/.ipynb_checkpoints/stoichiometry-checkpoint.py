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
    stoichiometry: tuple[dict[str, int]]

    @property
    def elements(self) -> set[str]:
        """Set of elements present in stoichiometry."""
        e = set()
        for s in self.stoichiometry:
            s = e.union(s.keys())
        return s

    # FIXME: Self only availabe in >=3.11
    def __add__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Extend underlying list of stoichiometries."""
        return Stoichiometry(self.stoichiometry + other.stoichiometry)

    def __or__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Inner product of underlying stoichiometries.

        Must not share elements with other stoichiometry."""
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only or stoichiometries of different elements!"
        s = ()
        for me, you in zip(self.stoichiometry, other.stoichiometry, strict=False):
            s += (me | you,)
        return Stoichiometry(s)

    def __mul__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Outer product of underlying stoichiometries.

        Must not share elements with other stoichiometry."""
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only multiply stoichiometries of different elements!"
        s = ()
        for me, you in product(self.stoichiometry, other.stoichiometry):
            s += (me | you,)
        return Stoichiometry(s)

    # Sequence Impl'
    def __getitem__(self, index: int) -> dict[str, int]:
        return self.stoichiometry[index]

    def __len__(self) -> int:
        return len(self.stoichiometry)

@as_function_node
def ElementInput(
    element: str,
    min_atoms: int = 1,
    max_atoms: int = 10,
    step_atoms: int = 1,
) -> Stoichiometry:
    stoichiometry = Stoichiometry(
        tuple({element: i} for i in range(min_atoms, max_atoms + 1, step_atoms))
    )
    return stoichiometry

@as_function_node("df")
def StoichiometryTable(stoichiometry: Stoichiometry) -> pd.DataFrame:
    return pd.DataFrame(stoichiometry.stoichiometry)


@as_function_node("filtered")
def FilterSize(
    elements: Stoichiometry,
    min_atoms: Optional[int] = 0,
    max_atoms: Optional[int] = 10,
):
    """Filter an Elements object by size.

    Args:
        min (int): new object has at least this number of atoms
        max (int): new object has at most this number of atoms

    Returns:
        Elements: filtered object
    """
    import math
    if max_atoms is None:
        max_atoms = math.inf
    return Stoichiometry(tuple(s for s in elements
                                if min_atoms <= sum(s.values()) <= max_atoms ))

def _atoms_to_structures_df(
    structures: list[Atoms],
    spacegroups: list[int],
    calc_type: str = ""
) -> pd.DataFrame:
    rows = []

    for i, (atoms, spg) in enumerate(zip(structures, spacegroups)):

        # Safe extraction (calculator may not exist)
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
            "energy": energy,              # float
            "forces": forces,              # (N, 3) np.ndarray
            "stress": stress,              # (6,) or (3,3)
        })

    return pd.DataFrame(rows)

def _structures_df_to_atoms(df: pd.DataFrame) -> list[Atoms]:

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
        spacegroups: list[int] | tuple[int,...] | None = None,
        max_atoms: int = 10,
        max_structures: Optional[int] = 50,
        engine: OutputEngine = None,
        store: bool = False
) -> pd.DataFrame:
    """
    Create symmetric random structures.

    Args:
        elements (Elements): list of compositions per structure
        spacegroups (list of int): which space groups to generate
        max_atoms (int): do not generate structures larger than this
        max_structures (int): generate at most this many structures
    Returns:
        list of Atoms: generated structures
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
        spacegroups = list(range(1,231))
    if max_structures == "":
        max_structures = math.inf

    structures = []
    spg_numbers = []
    with catch_warnings(category=UserWarning, action='ignore'):
        for stoich in (bar := tqdm(elements)):
            elements, num_atomss = zip(*stoich.items())
            stoich_str = "".join(f"{s}{n}" for s, n in zip(elements, num_atomss))
            bar.set_description(stoich_str)
            for s in pyxtal(spacegroups, elements, num_atomss):
                atoms = s["atoms"].copy()
                atoms.calc = engine.calculator
                structures.append(atoms)
                spg_numbers.append(get_symmetry(atoms).info["number"])
            if len(structures) > max_structures:
                print("Maximum number of structures reached! Ending the generation..")
                structures = structures[:max_structures]
                spg_numbers = spg_numbers[:max_structures]
                break
    for structure in tqdm(structures, desc="Spacegroup Calculations"):
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
    df_structures = _atoms_to_structures_df(structures = structures, spacegroups = spg_numbers)
    return df_structures
