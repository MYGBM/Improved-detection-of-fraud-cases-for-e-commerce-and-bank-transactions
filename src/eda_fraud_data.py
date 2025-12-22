import pandas as pd

class Eda:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Eda class with a pandas DataFrame.
        """
        self.df = df
        
    def load_data(filepath: str) -> pd.DataFrame:
        """
        Helper function to load data from a CSV file.
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully from {filepath}")
            return df
        except FileNotFoundError:
            print(f"File not found at {filepath}")
    
    def basic_exploration(self):
        """
        Performs basic data exploration steps: info, shape, describe, and head.
        Matches steps: data.info(), data.shape, data.describe(), data.head()
        """
        print("--- Data Info ---")
        print(self.df.info())
        print("\n--- Data Shape ---")
        print(self.df.shape)
        print("\n--- Data Description ---")
        print(self.df.describe())
        print("\n--- First 5 Rows ---")
        display(self.df.head()) if 'display' in globals() else print(self.df.head())

    def check_missing_values(self):
        """
        Checks for missing values in the dataset.
        Matches steps: data.isnull().sum(), data.isna().sum()
        """
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        print(missing)
        if missing.sum() == 0:
            print("No missing values found.")

    def check_unique_identifiers(self, column_name='user_id'):
        """
        Checks for uniqueness in a specific identifier column.
        Matches step: data["user_id"].nunique()
        """
        print(f"\n--- Checking Uniqueness for {column_name} ---")
        unique_count = self.df[column_name].nunique()
        total_rows = len(self.df)
        print(f"Unique count: {unique_count}")
        print(f"Total rows: {total_rows}")
        
        if unique_count == total_rows:
            print(f"All values in '{column_name}' are unique (no duplicates).")
        else:
            print(f"There are {total_rows - unique_count} duplicate values in '{column_name}'.")

    def convert_datetypes(self, columns: list) -> pd.DataFrame:
        """
        Converts date columns to appropriate date types
        example: columns = ['signup_time', 'purchase_time']
        Matches steps:
        - signup_time -> datetime
        - purchase_time -> datetime
        """
        print("\n--- Converting Data Types ---")
        try:
            for column in columns:
            # Convert timestamps
                self.df[column] = pd.to_datetime(self.df[column])
                self.df[column] = pd.to_datetime(self.df[column])          
            print("Conversion successful.")
            print(self.df.dtypes)
        except Exception as e:
            print(f"Error during conversion: {e}")
        
        return self.df

