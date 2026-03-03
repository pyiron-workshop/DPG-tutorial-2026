from core import as_function_node


@as_function_node
def Ace(potential_file, use_symmetry:bool=True):
    """
    Create an ACE (Atomic Cluster Expansion) atomistic engine.

    This node initializes a PyACE calculator from a potential file and wraps
    it in an ``OutputEngine`` object for use in atomistic simulation workflows.
    ACE potentials offer a systematically improvable, body-order expanded
    representation of the potential energy surface.

    Typical use cases include:
    - Energy, force, and stress evaluation with ACE machine-learning potentials
    - Structure relaxation
    - Molecular dynamics simulations

    Args:
        potential_file (str or path-like):
            Path to the ACE potential file (e.g. a ``.yaml`` or ``.acecoefficients``
            file) used to initialize the ``PyACECalculator``.
        use_symmetry (bool):
            Whether to exploit crystal symmetry during evaluation. Defaults
            to ``True``.

    Returns:
        OutputEngine:
            Engine object containing the ACE ASE-compatible calculator.
    """
    from pyace import PyACECalculator
    calc = PyACECalculator(potential_file)
    from logging import ERROR
    from pyiron_snippets.logger import logger

    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=calc)
    logger.setLevel(ERROR)
    return out
