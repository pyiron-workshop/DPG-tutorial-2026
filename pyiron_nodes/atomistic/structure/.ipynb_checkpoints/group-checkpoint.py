from ase.atoms import Atoms
import logging
import pandas as pd
from core import as_function_node
from typing import Optional 

@as_function_node
def CombineStructures(
        spacegroups: Optional[pd.DataFrame],
        volume_relax: Optional[pd.DataFrame],
        full_relax: Optional[pd.DataFrame],
        rattle: Optional[pd.DataFrame],
        stretch: Optional[pd.DataFrame],
        filename: Optional[str] = "data/Structures_Everything",
        store: bool = True,
) -> pd.DataFrame:
    """Combine individual structure sets into a full training set."""
    import os.path
    
    dfs = [spacegroups, volume_relax, full_relax, rattle, stretch]

    # keep only non-None and non-empty DataFrames
    dfs = [df for df in dfs if df is not None and not ""]

    if len(dfs) == 0:
        logging.warning(
            "Either no inputs given or all inputs are empty. "
            "Returning an empty DataFrame!"
        )    
    df_total = pd.concat(dfs, ignore_index=True)
    df_total["name"] = df_total["name"].str.replace(r"_structure_\d+$", "", regex=True) + "_structure_"+ pd.Series(range(len(df_total))).astype(str)
    if filename != "":
        if not filename.endswith("pckl.gz"):
            filename += ".pckl.gz"
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        df_total.to_pickle(filename, compression = "gzip")
    return df_total