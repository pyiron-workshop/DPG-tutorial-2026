from core import as_function_node
from functools import lru_cache


@as_function_node("engine")
@lru_cache
def GRACE(model: str = "GRACE-FS-OAM"):
    """
    Create and cache a GRACE atomistic engine.

    This node initializes a Graph Atomic Cluster Expansion (GRACE)
    calculator and wraps it in an ``OutputEngine`` object for use in
    atomistic simulation workflows. The engine can be reused across
    multiple nodes and workflow steps.

    The function is cached to avoid repeated initialization of the
    underlying model, which can be computationally expensive.

    Typical use cases include:
    - Energy, force, and stress evaluation
    - Structure relaxation
    - Data generation for machine-learning potentials

    Args:
        model (str):
            Identifier of the pretrained GRACE model to load
            (e.g. ``"GRACE-FS-OAM"``).

    Returns:
        OutputEngine:
            Engine object containing the GRACE ASE-compatible calculator.
    """
    from tensorpotential.calculator import grace_fm
    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=grace_fm(model))
    return out