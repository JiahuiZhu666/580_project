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

def split_data(data: pd.DataFrame, date_split: str) -> pd.DataFrame:
    """
    Splits the data into training and testing sets based on a date.

    Parameters:
    data (DataFrame): The data to split.
    date——split (str): The date to split the data on.

    Returns:
    DataFrame: The training and testing data.
    """
    train = data[:date_split]
    test = data[date_split:]
    return train, test