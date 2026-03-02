from __future__ import annotations
from typing import Literal, Optional

import numpy as np
from ase import Atoms as _Atoms

from core import as_function_node


@as_function_node("plot")
def Plot3d(
    structure: _Atoms,
    camera: str = "orthographic",
    particle_size: float = 1.0,
    background: Literal["white", "black"] = "white",
    select_atoms: Optional[np.ndarray | list] = None,
    view_plane: Optional[list] = None,
    distance_from_camera: Optional[float] = 1.0,
):
    """
    Display atomistic structure (ase.Atoms) using nglview.

    Task
    ----
    Visualise a static atomic structure, e.g., after building a bulk cell,
    creating a surface slab, or after a geometry optimisation. This node is
    typically used when the user wants to inspect the geometry, defects, or
    surface features directly in a Jupyter notebook.

    Parameters
    ----------
    structure: ase.Atoms
        The atomic structure to visualise.
    camera: str, optional
        Camera mode, either "orthographic" or "perspective".
    particle_size: float, optional
        Size of the rendered atoms.
    background: {"white", "black"}, optional
        Background colour of the view.
    select_atoms: np.ndarray or list, optional
        Indices of atoms to highlight.
    view_plane: list, optional
        Plane normal for the view.
    distance_from_camera: float, optional
        Distance of the camera from the structure.
    """

    if view_plane is None:
        view_plane = [1, 1, 1]

    from structuretoolkit.visualize import plot3d
    return plot3d(
        structure,
        camera=camera,
        particle_size=particle_size,
        background=background,
        select_atoms=select_atoms,
        view_plane=view_plane,
        distance_from_camera=distance_from_camera,
    )
