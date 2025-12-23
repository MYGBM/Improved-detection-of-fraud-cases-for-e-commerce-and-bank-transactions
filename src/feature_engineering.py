import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEngineer class with a pandas DataFrame.
        """
        self.df = df
        
    @staticmethod
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
            return None

    def add_user_transaction_count(self) -> pd.DataFrame:
        """
        Adds a feature 'user_transaction_count' counting the number of transactions per user.
        """
        print("\n--- Adding User Transaction Count ---")
        if 'user_id' not in self.df.columns:
            raise ValueError("Column 'user_id' not found in DataFrame")
            
        # Count transactions per user
        user_counts = self.df['user_id'].value_counts()
        self.df['user_transaction_count'] = self.df['user_id'].map(user_counts)
        
        print("Feature 'user_transaction_count' added.")
        return self.df

    def add_time_features(self) -> pd.DataFrame:
        """
        Adds time-based features:
        - purchase_time_hour: Hour of the day from purchase_time
        - purchase_time_day_of_week: Day of the week from purchase_time
        
        """
        print("\n--- Adding Time Features ---")
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df['purchase_time']):
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
            
        self.df['purchase_time_hour'] = self.df['purchase_time'].dt.hour
        self.df['purchase_time_day_of_week'] = self.df['purchase_time'].dt.dayofweek
        
        print("Features 'purchase_time_hour' and 'purchase_time_day_of_week' added.")
        return self.df

    def add_time_since_signup(self) -> pd.DataFrame:
        """
        Calculates the time difference between signup_time and purchase_time.
        """
        print("\n--- Adding Time Since Signup ---")
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df['purchase_time']):
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
        if not pd.api.types.is_datetime64_any_dtype(self.df['signup_time']):
            self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
            
        # Calculate difference in seconds (more granular than hours/days)
        self.df['purchase_time_since_signup_seconds'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds()
        
        # Convert to hours for easier interpretation if needed, but seconds is good for models
        self.df['purchase_time_since_signup_hours'] = self.df['purchase_time_since_signup_seconds'] / 3600
        
        print("Feature 'purchase_time_since_signup_seconds' (in seconds) added.")
        print("Feature 'purchase_time_since_signup_hours' (in hours) added.")
        return self.df
    
    def drop_columns(self, columns: list) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.
        """
        print(f"\n--- Dropping Columns: {columns} ---")
        self.df.drop(columns=columns, inplace=True)
        print(f"Columns {columns} dropped.")
        return self.df
