from core import as_function_node

@as_function_node
def EAM(
    potential_name: str
):
    """
    Create a workflow-ready “potential engine” node that represents an EAM potential by name.

    **Scientific purpose**
    Provide an interatomic potential handle (Embedded Atom Method, EAM) that downstream atomistic
    simulation nodes can consume (e.g., LAMMPS-based MD, structure relaxation, or Calphy free-energy
    calculations). In practice, this node stores the *identifier* of a potential from the pyiron/IPRPy
    database so it can be resolved consistently later.

    **Required inputs**
    - ``potential_name``: string name/ID of the potential (typically one entry returned by
      ``ListPotentials`` / the pyiron potential database).

    **Typical use-cases**
    * Select an EAM potential for a given element/alloy system and pass it into Calphy nodes
      (solid/liquid free energy, temperature sweeps).
    * Build reproducible workflows where the chosen potential is an explicit graph node.

    Returns
    -------
    OutputEngine
        A ``pyiron_nodes.atomistic.engine.generic.OutputEngine`` whose ``calculator()`` returns
        the provided ``potential_name``. Downstream nodes may resolve this name via pyiron’s
        potential database (e.g., ``pyiron_lammps.potential.get_potential_by_name``).

    Notes
    -----
    This node does not validate that the potential exists or is compatible with the structure;
    use ``ListPotentials`` (or a direct lookup) for discovery/validation.
    """
    from logging import ERROR
    from pyiron_lammps.potential import get_potential_by_name
    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from pyiron_snippets.logger import logger
    
    # potential_dataframe = get_potential_by_name(potential_name)
    def eam_calc():
        potential_str = potential_name
        return potential_str
        
    engine = OutputEngine(calculator=eam_calc)

    logger.setLevel(ERROR)
    return engine