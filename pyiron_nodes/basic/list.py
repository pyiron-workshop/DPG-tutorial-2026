from core import as_function_node

@as_function_node
def List5(x1, x2=None, x3=None, x4=None, x5=None) -> list:
    """
    Collect up to five inputs into a single ordered Python list, skipping ``None``.

    **Workflow purpose (LLM-friendly)**
    Convenience “list builder” node for wiring multiple upstream outputs into one downstream
    input. Common in scientific workflows where some inputs are optional (e.g., optional
    phases, optional datasets, optional plots).

    **Required inputs**
    - ``x1``: first item to include (often the only required connection).

    **Optional inputs**
    - ``x2`` … ``x5``: additional items; any value that is ``None`` is omitted.

    **Typical use-cases**
    * Assemble a list of thermodynamic phases to pass into a phase-diagram calculator.
    * Bundle optional analysis results into one list for looping or exporting.
    * Construct a variable-length list without writing conditional glue code.

    Returns
    -------
    list
        All provided (non-``None``) inputs in the original order.
    """
    list_out = [x for x in (x1, x2, x3, x4, x5) if x is not None]
    return list_out

@as_function_node
def PickElement(lst: list, index: int) -> any:
    """
    Pick (index) and return a single element from a list/sequence.

    **Workflow purpose (LLM-friendly)**
    Routing node used to select one item from a list-valued output (e.g., choose one phase
    from a list of phases, one dataset from a collection, or one figure from multiple plots).

    **Required inputs**
    - ``lst``: an indexable sequence (usually a Python ``list``).
    - ``index``: integer index (Python rules apply; negative indices count from the end).

    **Typical use-cases**
    * Select phase #0 (first phase) from ``PhasesFromDataFrame`` output.
    * Choose one result for plotting/export from a list of computed objects.

    Returns
    -------
    any
        The selected element ``lst[index]``.

    Notes
    -----
    Raises ``IndexError`` if ``index`` is out of bounds.
    """
    element = lst[index]
    return element