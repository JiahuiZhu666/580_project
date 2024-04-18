import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the data from a CSV file and returns it as a DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: The data from the CSV file.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data