from typing import Optional

from ase import Atoms
from core import as_function_node

@as_function_node
def ListPotentials(
    structure: Atoms, 
    resource_path: Optional[str] = None
) -> list:
    """
    List interatomic potentials compatible with a given atomic structure (elements) for LAMMPS/pyiron.

    **Scientific purpose**
    Discover which parameterized interatomic potentials (e.g., EAM/MEAM/alloy potentials)
    are available for the chemical species present in an ASE `Atoms` object. This is a
    practical “model selection” step before running atomistic simulations such as MD,
    relaxation, or free-energy calculations (e.g., with Calphy).

    **Required inputs**
    - ``structure``: ASE ``Atoms``; the element symbols in this structure determine which
      potentials are considered compatible.

    **Optional inputs**
    - ``resource_path``: path to the IPRPy/pyiron potential database directory. If omitted,
      defaults to ``$CONDA_PREFIX/share/iprpy``.

    **Typical use-cases**
    * Find all potentials that support (Fe), (Al), or multi-component systems (e.g., Ni–Al).
    * Provide a dropdown/auto-complete list of potential names for subsequent workflow nodes.
    * Validate that a chosen potential exists before launching LAMMPS/Calphy jobs.

    Returns
    -------
    list[str]
        Names/identifiers of potentials reported by ``pyiron_lammps.potential.view_potentials``
        as compatible with the given structure.
    """
    import os
    from pyiron_lammps.potential import view_potentials

    if resource_path is None:
        resource_path = os.path.join(os.environ["CONDA_PREFIX"], "share", "iprpy")

    potentials = list(view_potentials(structure, resource_path=resource_path)["Name"].values)

    return potentials