import pandas as pd

def modeler(date: str, data: list[dict[str, str]]) -> None:
    """
    Processes and saves anime data to a CSV file.

    -----
    Parameters:
    - date (str): The date string used to name the CSV file.
    - data (List[Dict[str, str]]): A list of dictionaries containing anime data.

    -----
    The function converts the list of dictionaries to a DataFrame, removes duplicate entries,
    and saves the DataFrame to a CSV file in the 'data/processed' directory.
    """
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    file_path = f'data/processed/AnimeData_{date}.csv'
    df.to_csv(file_path, index=False)
    print(f'Data saved to {file_path}')