from core import as_function_node


@as_function_node
def ListPotentials(structure):
    from pyiron_lammps.potential import view_potentials

    potentials = list(view_potentials(structure)["Name"].values)
    return potentials
