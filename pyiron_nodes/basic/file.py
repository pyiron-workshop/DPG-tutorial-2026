from core import as_function_node


@as_function_node("csv")
def ReadCSV(filename: str, header: list = [0, 1], decimal: str = ",", delimiter: str = ";"):
    import pandas as pd
    return pd.read_csv(filename, delimiter=delimiter, header=header, decimal=decimal)


@as_function_node("df")
def ReadDataFrame(filename: str, compression: str = None):
    import pandas as pd

    return pd.read_pickle(filename, compression=compression)

import pandas as pd
@as_function_node
def DataframeToList(dataframe:pd.DataFrame, column:str='structure'):
    if column:
        list_of_values = dataframe[column].tolist()
    else:
        print("Give column name")
        list_of_values = None
    return list_of_values
