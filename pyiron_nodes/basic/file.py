from core import as_function_node
import pandas as pd

@as_function_node("csv")
def ReadCSV(filename: str, header: list = [0, 1], decimal: str = ",", delimiter: str = ";"):
    """
    Read a CSV file into a pandas DataFrame.

    This node loads a delimited text file using pandas, supporting
    multi-level column headers and locale-specific number formatting.
    Commonly used to ingest benchmarking results or tabular data exported
    from spreadsheet tools or other workflows.

    Typical use cases include:
    - Loading interatomic potential benchmark results stored as CSV
    - Reading tabular property data with multi-level column headers
    - Importing data with European-style decimal and delimiter conventions

    Args:
        filename (str):
            Path to the CSV file to read.
        header (list):
            Row number(s) to use as column headers, supporting multi-level
            (hierarchical) column indices. Defaults to ``[0, 1]``.
        decimal (str):
            Character used as the decimal separator in the file.
            Defaults to ``","`` (European convention).
        delimiter (str):
            Character used to separate fields in the file.
            Defaults to ``";"`` (European CSV convention).

    Returns:
        pandas.DataFrame:
            DataFrame containing the parsed contents of the CSV file.
    """

    import pandas as pd
    return pd.read_csv(filename, delimiter=delimiter, header=header, decimal=decimal)


@as_function_node("df")
def ReadDataFrame(filename: str, compression: str = None):
    """
    Load a serialized pandas DataFrame from a pickle file.

    This node deserializes a DataFrame that was previously saved using
    ``pandas.DataFrame.to_pickle``, optionally decompressing it on the fly.
    Useful for efficiently reloading large or complex DataFrames — including
    those with nested structures or non-CSV-serializable dtypes — between
    workflow steps.

    Typical use cases include:
    - Reloading cached workflow results or intermediate computation outputs
    - Loading nested DataFrames containing simulation results (e.g. surface
      stress, elastic moduli, stacking fault energies)
    - Restoring DataFrames with multi-index columns or object-typed entries

    Args:
        filename (str):
            Path to the pickle file to load.
        compression (str or None):
            Compression scheme used when the file was saved (e.g.
            ``"gzip"``, ``"bz2"``, ``"zip"``, ``"xz"``). Pass ``None``
            for uncompressed files. Defaults to ``None``.

    Returns:
        pandas.DataFrame:
            Deserialized DataFrame loaded from the pickle file.
    """
    import pandas as pd

    return pd.read_pickle(filename, compression=compression)

@as_function_node
def DataframeToList(dataframe:pd.DataFrame, column:str='structure'):
    """
    Extract a single DataFrame column as a Python list.

    This node pulls a specified column out of a DataFrame and returns its
    values as a plain list, making it straightforward to feed tabular data
    into downstream nodes that expect list inputs — for example, passing a
    list of ASE ``Atoms`` structures into a simulation engine.

    Typical use cases include:
    - Extracting a list of atomic structures from a structures DataFrame
    - Feeding a column of file paths, identifiers, or property values into
      a mapping or iteration node
    - Unpacking nested DataFrame entries for sequential processing

    Args:
        dataframe (pandas.DataFrame):
            Source DataFrame containing the column to extract.
        column (str):
            Name of the column to convert to a list. Defaults to
            ``"structure"``.

    Returns:
        list or None:
            List of values from the specified column, or ``None`` if no
            column name was provided.
    """
    
    if column:
        list_of_values = dataframe[column].tolist()
    else:
        print("Give column name")
        list_of_values = None
    return list_of_values
