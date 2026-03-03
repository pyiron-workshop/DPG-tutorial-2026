from core import as_function_node
from typing import Optional
import pandas as pd
import numpy as np


@as_function_node
def ReadPickledDatasetAsDataframe(
    file_path: str = "", compression: Optional[str] = None
):
    """
    Load a pickled atomistic dataset into a standardized pandas DataFrame.

    This node reads a pickled dataset containing atomistic simulation data
    and normalizes column names and formats to ensure compatibility with
    downstream fitting, prediction, and analysis nodes.

    The function performs several consistency checks and conversions:
    - Ensures ASE ``Atoms`` objects are stored under the ``ase_atoms`` column
    - Normalizes atom-count column naming
    - Ensures a corrected energy column is present
    - Infers periodic boundary conditions if missing

    Args:
        file_path (str):
            Path to the pickled dataset file.
        compression (str, optional):
            Compression type used for the pickle file (e.g. ``"gzip"``).

    Returns:
        pandas.DataFrame:
            Cleaned and standardized dataset ready for atomistic
            machine-learning workflows.
    """
    from ase.atoms import Atoms as aseAtoms

    df = pd.read_pickle(file_path, compression=compression)

    # Atoms check
    if "atoms" in df.columns:
        at = df.iloc[0]["atoms"]
        # Checking that the elements themselves have the correct atoms format
        if isinstance(at, aseAtoms):
            df.rename(columns={"atoms": "ase_atoms"}, inplace=True)
    elif "ase_atoms" not in df.columns:
        raise ValueError(
            "DataFrame should contain 'atoms' or 'ase_atoms' (ASE atoms) columns"
        )

    # NUMBER OF ATOMS check
    if "NUMBER_OF_ATOMS" not in df.columns and "number_of_atoms" in df.columns:
        df.rename(columns={"number_of_atoms": "NUMBER_OF_ATOMS"}, inplace=True)

    df["NUMBER_OF_ATOMS"] = df["NUMBER_OF_ATOMS"].astype(int)

    # energy corrected check
    if "energy_corrected" not in df.columns and "energy" in df.columns:
        df.rename(columns={"energy": "energy_corrected"}, inplace=True)

    if "pbc" not in df.columns:
        df["pbc"] = df["ase_atoms"].map(lambda atoms: np.all(atoms.pbc))

    return df


@as_function_node
def SplitTrainingAndTesting(
    data_df: pd.DataFrame,
    training_frac: float = 0.5,
    random_state: int = 42
):
    """
    Split an atomistic dataset into training and testing subsets.

    This node randomly partitions a DataFrame into training and testing
    datasets according to a specified fraction. It is commonly used
    in supervised learning workflows for interatomic potential fitting
    and validation.

    Basic sanity checks are applied to ensure a non-zero and valid
    training fraction.

    Args:
        data_df (pandas.DataFrame):
            Input dataset containing atomistic structures and reference data.
        training_frac (float):
            Fraction of the dataset to use for training (between 0 and 1).
        random_state (int):
            Random seed for reproducible shuffling and sampling.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]:
            Training and testing DataFrames.
    """
    if isinstance(training_frac, float):
        training_frac = np.abs(training_frac)

    if training_frac > 1:
        print("""
            Can't have the training dataset more than 100 % of the dataset
            Setting the value to 100%
            """)
        training_frac = 1
    elif training_frac == 0:
        print("Can'fit with no training dataset\nSetting the value to 1%")
        training_frac = 0.01

    df_training = data_df.sample(frac=training_frac, random_state=random_state)
    df_testing = data_df.loc[(i for i in data_df.index if i not in df_training.index)]

    return df_training, df_testing