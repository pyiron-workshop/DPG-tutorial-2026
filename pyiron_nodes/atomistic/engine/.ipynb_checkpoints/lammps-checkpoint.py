from ase import Atoms

from core import as_function_node

@as_function_node
def ListPotentials(
    structure: Atoms, 
    resource_path: str=None
) -> list:
    
    import os
    from pyiron_lammps.potential import view_potentials

    if resource_path is None:
        resource_path = os.path.join(os.environ["CONDA_PREFIX"], "share", "iprpy")

    potentials = list(view_potentials(structure, resource_path=resource_path)["Name"].values)

    return potentials