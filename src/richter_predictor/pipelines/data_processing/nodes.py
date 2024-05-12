import pandas as pd

def load_raw_data(train_values: pd.DataFrame, test_values: pd.DataFrame) -> pd.DataFrame:
    """
    Load raw data from a source and return it as a pandas DataFrame.

    Returns:
    --------
        pd.DataFrame: The raw data as a pandas DataFrame.
    """

    df = pd.concat([
        train_values.assign(is_train=True),
        test_values.assign(is_train=False)
    ])

    df = df.drop('building_id', axis=1)

    return df

def define_categorical_columns(df: pd.DataFrame, non_categorical_columns: list) -> list:
    """
    Define the categorical columns in the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        list: A list of column names that are categorical.
    """
    
    categorical_columns = [col for col in df.drop(non_categorical_columns, axis=1).columns]
    
    return categorical_columns
