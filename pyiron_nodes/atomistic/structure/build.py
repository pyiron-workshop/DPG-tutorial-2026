from typing import Optional, Literal

from ase.atoms import Atoms
from ase.build import bulk
from core import as_function_node

@as_function_node("structure")
def Bulk(
    name: str,
    crystalstructure: Optional[
        Literal["fcc", "bcc", "hcp", "diamond", "rocksalt"]
    ] = None,
    a: Optional[float] = None,
    c: Optional[float] = None,
    u: Optional[float] = None,
    orthorhombic: bool = False,
    cubic: bool = False,
) -> Atoms:
    """
    Create a bulk crystal structure.

    **Scientific purpose**
    Generate a pristine bulk unit cell for a given element and crystal lattice
    (e.g. fcc Al, bcc Fe, hcp Mg). This is the typical starting point for defect
    studies, molecular‑dynamics simulations, or high‑throughput materials screening.

    **Required inputs**
    - ``name``: Chemical symbol of the element (e.g. ``"Al"``).
    - ``crystalstructure``: One of ``"fcc"``, ``"bcc"``, ``"hcp"``, ``"diamond"``,
      ``"rocksalt"``; if omitted the default of the underlying factory is used.
    - ``a``, ``c``, ``c_over_a``, ``u``: Optional lattice parameters or internal
      coordinates.
    - ``orthorhombic`` / ``cubic``: Force the cell shape to be orthorhombic or cubic.

    **Typical use‑cases**
    * Building a bulk material before inserting vacancies, interstitials, or
      surfaces.
    * Preparing input structures for relaxation or MD runs.
    * Generating a library of bulk cells for high‑throughput workflows.

    Returns
    -------
    Node returns an Atoms object from ``pyiron_atomistics._StructureFactory().bulk`` compatible
    with ASE/pyiron.
    """
    return bulk(
        name=name,
        crystalstructure=crystalstructure,
        a=a,
        c=c,
        u=u,
        orthorhombic=orthorhombic,
        cubic=cubic,
    )

def _parse_formula(formula):
    import re
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    elements = [el for el, num in parts]
    return elements

def _generate_CaMg_ortho():
    from ase import Atoms
    import numpy as np
    # Lattice vectors
    cell = np.array([
        [3.692641, 0.0, 0.0000000000000002],
        [-0.0000000000000004, 5.809565, 0.0000000000000004],
        [0.0, 0.0, 5.9892969999999996]
    ])

    # Element symbols
    symbols = ['Ca', 'Ca', 'Mg', 'Mg']

    # Fractional coordinates
    scaled_positions = np.array([
        [0.0000000000000000, 0.0000000000000000, 0.0932210000000000],
        [0.5000000000000000, 0.5000000000000000, 0.9067790000000000],
        [0.5000000000000000, 0.0000000000000000, 0.5849690000000000],
        [0.0000000000000000, 0.5000000000000000, 0.4150310000000000]
    ])

    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)

    return atoms
    
@as_function_node
def Bulk2(
    name: str,
    crystalstructure: Optional[
        Literal["fcc", "bcc", "hcp", "diamond", "rocksalt"]
    ] = None,
    a: Optional[float] = None,
    c: Optional[float] = None,
    u: Optional[float] = None,
    orthorhombic: bool = False,
    cubic: bool = True,    #<--- Changed to True
) -> Atoms:
    species = _parse_formula(name)
    if len(species)==1:
        structure = bulk(
                            name=name,
                            crystalstructure=crystalstructure,
                            a=a,
                            c=c,
                            u=u,
                            orthorhombic=orthorhombic,
                            cubic=cubic,
                        )
    else:
        print("Building Orthorhombic CaMg")
        structure = _generate_CaMg_ortho()
        
    return structure
