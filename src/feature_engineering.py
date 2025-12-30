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

# ...existing code...
    def normalize_features(self, columns: list, method: str = 'standard') -> pd.DataFrame:
        """
        Normalizes or scales numerical features.
        
        Parameters:
        - columns: List of column names to scale.
        - method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        print(f"\n--- Normalizing Features ({method}) ---")
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
            
        # Check if columns exist
        valid_cols = [col for col in columns if col in self.df.columns]
        
        if valid_cols:
            self.df[valid_cols] = scaler.fit_transform(self.df[valid_cols])
            print(f"Scaled columns: {valid_cols}")
        else:
            print("No valid columns found to scale.")
            
        return self.df

    def apply_log_transform(self, columns: list) -> pd.DataFrame:
        """
        Applies log transformation (log1p) to specified columns to handle skewness.
        """
        import numpy as np
        print(f"\n--- Applying Log Transformation ---")
        valid_cols = [col for col in columns if col in self.df.columns]
        
        if not valid_cols:
            print("No valid columns found to log transform.")
            return self.df

        for col in valid_cols:
            # Ensure no negative values for log
            if (self.df[col] < 0).any():
                 print(f"Warning: Column {col} contains negative values. Skipping log transform.")
                 continue
            
            self.df[col] = np.log1p(self.df[col])
            print(f"Log transformed: {col}")
            
        return self.df

    def encode_categorical_features(self, columns: list, method: str = 'onehot') -> pd.DataFrame:
        """
        Encodes categorical features.
        
        Parameters:
        - columns: List of column names to encode.
        - method: 'onehot', 'label', or 'frequency'.
        """
        print(f"\n--- Encoding Categorical Features ({method}) ---")
        
        valid_cols = [col for col in columns if col in self.df.columns]
        
        if not valid_cols:
            print("No valid columns found to encode.")
            return self.df

        if method == 'onehot':
            # pd.get_dummies is easier for DataFrames than sklearn OneHotEncoder
            self.df = pd.get_dummies(self.df, columns=valid_cols, drop_first=True,dtype=int)
            print(f"One-hot encoded: {valid_cols}")
            
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in valid_cols:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
            print(f"Label encoded: {valid_cols}")

        elif method == 'frequency':
            for col in valid_cols:
                freq_encoding = self.df[col].value_counts(normalize=True)
                self.df[col] = self.df[col].map(freq_encoding)
            print(f"Frequency encoded: {valid_cols}")
            
        return self.df
    
    def cyclical_encode(self, columns: list, max_vals: dict) -> pd.DataFrame:
        """
        Applies cyclical encoding (sin/cos) to time-based features.
        
        Parameters:
        - columns: List of columns to encode.
        - max_vals: Dictionary mapping column name to its maximum cycle value 
                    (e.g., {'hour': 24, 'day': 7}).
        """
        import numpy as np
        print("\n--- Applying Cyclical Encoding ---")
        
        for col in columns:
            if col not in self.df.columns:
                print(f"Column {col} not found.")
                continue
            
            max_val = max_vals.get(col)
            if max_val is None:
                print(f"Max value for {col} not provided.")
                continue

            # Create Sin and Cos features
            self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / max_val)
            self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / max_val)
            
            print(f"Encoded {col} -> {col}_sin, {col}_cos")
            
            # Drop the original column as it's no longer needed
            self.df.drop(columns=[col], inplace=True)
            
        return self.df
    
    def check_feature_stats(self, columns: list, stage: str = "Current") -> None:
        """
        Prints descriptive statistics (mean, std, min, max) for specific columns
        to help evaluate the effect of transformations.
        """
        print(f"\n--- Feature Statistics ({stage}) ---")
        valid_cols = [col for col in columns if col in self.df.columns]
        
        if not valid_cols:
            print("No valid columns found.")
            return

        stats = self.df[valid_cols].describe().T[['mean', 'std', 'min', 'max']]
        print(stats)
        
    def plot_pre_post_distribution(self, original_df: pd.DataFrame, columns: list, isBefore: bool = True) -> None:
        """
        Plots the distribution of features before and after transformation.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("\n--- Plotting Pre vs Post Distributions ---")
        
        
        for col in columns:
            if col not in self.df.columns or col not in original_df.columns:
                continue

            plt.figure(figsize=(12, 5))
            if isBefore:
                # Plot Original (Before)
                plt.subplot(1, 2, 1)
                sns.histplot(original_df[col], kde=True, color='blue', bins=30)
                plt.title(f'Before: {col} (Original)')
                plt.xlabel('Value')
            else:
                # Plot Current (After)
                plt.subplot(1, 2, 2)
                sns.histplot(self.df[col], kde=True, color='green', bins=30)
                plt.title(f'After: {col} (Transformed)')
                plt.xlabel('Value (Scaled)')

                plt.tight_layout()
                plt.show()
