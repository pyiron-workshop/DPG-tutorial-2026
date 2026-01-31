from pyiron_core import as_function_node
import pandas as pd
import numpy as np


@as_function_node("product")
def Multiply(x, y):
    return x * y


@as_function_node("matrix")
def SliceArray(matrix, indices):
    return matrix[indices]


@as_function_node("vector")
def GetVector(
    df: pd.DataFrame,
    indices,
    scale_energy_per_atom: bool = False,
):
    import numpy as np

    vec = df.energy_corrected
    if scale_energy_per_atom:
        vec /= df.NUMBER_OF_ATOMS

    forces_vec = []
    for f in df.forces.apply(lambda x: x.flatten()):
        forces_vec += list(f)
    vec = np.append(vec, forces_vec)
    return vec[indices]


@as_function_node
def MinMaxIndices(
    df: pd.DataFrame,
    i_min: int = 0,
    i_max: int = None,
    energy_only: bool = False,
):
    num_structures = len(df)
    num_atoms = np.sum(df.NUMBER_OF_ATOMS)

    indices = np.arange(num_structures + 3 * num_atoms)
    if i_max is None or i_max == "":
        i_max = num_structures
    energies = indices[i_min:i_max]
    forces = indices[num_atoms + 3 * i_min : num_atoms + 3 * i_max]
    if energy_only:
        indices = energies
    else:
        indices = np.append(energies, forces, axis=0)
    return indices


@as_function_node("linspace")
def Linspace(
    x_min: float = 0,
    x_max: float = 1,
    num_points: int = 50,
    endpoint: bool = True,
):
    return np.linspace(x_min, x_max, num_points, endpoint=endpoint)