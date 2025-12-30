import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

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




    def eda_univariate_distribution(self, column: str, bins: int = 30, figsize: tuple = (8, 5),
                    kde: bool = False, log_scale: bool = False) -> None:
        """Plot a histogram for a dataframe column to show its distribution.

        Parameters
        - column: name of the column to plot
        - bins: number of histogram bins
        - figsize: figure size (width, height)
        - kde: whether to overlay a kernel density estimate (uses seaborn if available)
        - log_scale: whether to use a log scale on the x-axis

        The function displays the plot (suitable for Jupyter notebooks).
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        series = self.df[column].dropna()

        plt.figure(figsize=figsize)

        # Prefer seaborn if available for nicer defaults and optional KDE
        try:
            import seaborn as sns  # type: ignore
            sns.histplot(series, bins=bins, kde=kde)
        except Exception:
            # Fallback to matplotlib
            plt.hist(series, bins=bins)

        if log_scale:
            try:
                plt.xscale('log')
            except Exception:
                pass

        plt.title(f"Distribution of '{column}'")
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    def eda_bivariate_categorical(self, feature: str, target: str = 'class', figsize: tuple = (12, 6)):
        """
        Calculates and plots the proportion of fraud (target=1) for each category in the feature,
        along with the transaction count to provide context on sample size.
        """
        if feature not in self.df.columns:
            raise ValueError(f"Column '{feature}' not found in DataFrame")
            
        # Calculate count and mean (fraud rate)
        stats = self.df.groupby(feature)[target].agg(['count', 'mean']).sort_values('mean', ascending=False)
        print(f"\n--- Statistics for {feature} ---")
        print(stats)
        
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot Count on primary y-axis (Bar)
        try:
            import seaborn as sns
            sns.barplot(x=stats.index, y=stats['count'], ax=ax1, color='skyblue', alpha=0.6)
        except ImportError:
            stats['count'].plot(kind='bar', ax=ax1, color='skyblue', alpha=0.6)

        ax1.set_ylabel('Transaction Count', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax1.set_xlabel(feature)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        # Plot Fraud Rate on secondary y-axis (Line)
        ax2 = ax1.twinx()
        # We use a line plot for the rate to distinguish it
        ax2.plot(range(len(stats)), stats['mean'], color='red', marker='o', linewidth=2, label='Fraud Rate')
        ax2.set_ylabel('Fraud Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f"Transaction Count and Fraud Rate by {feature}")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return stats

    def eda_bivariate_continuous(self, feature: str, target: str = 'class', bins: int = 10, figsize: tuple = (12, 6)):
        """
        Bins a continuous feature and plots the fraud rate and count for each bin to show trends.
        """
        if feature not in self.df.columns:
            raise ValueError(f"Column '{feature}' not found in DataFrame")
            
        # Create bins
        try:
            # Use cut for equal-width bins to see absolute value trends
            binned_series = pd.cut(self.df[feature], bins=bins)
            stats = self.df.groupby(binned_series)[target].agg(['count', 'mean'])
        except Exception as e:
            print(f"Error binning data with cut: {e}. Trying qcut...")
            binned_series = pd.qcut(self.df[feature], q=bins, duplicates='drop')
            stats = self.df.groupby(binned_series)[target].agg(['count', 'mean'])

        print(f"\n--- Statistics for {feature} (Binned) ---")
        print(stats)

        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot Count (Bar)
        x_labels = stats.index.astype(str)
        try:
            import seaborn as sns
            sns.barplot(x=x_labels, y=stats['count'], ax=ax1, color='lightgreen', alpha=0.6)
        except ImportError:
            stats['count'].plot(kind='bar', ax=ax1, color='lightgreen', alpha=0.6)

        ax1.set_ylabel('Transaction Count', color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_xlabel(f"{feature} Range")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        # Plot Rate (Line)
        ax2 = ax1.twinx()
        ax2.plot(range(len(stats)), stats['mean'], color='darkblue', marker='o', linewidth=2, label='Fraud Rate')
        ax2.set_ylabel('Fraud Rate', color='darkblue')
        ax2.tick_params(axis='y', labelcolor='darkblue')

        plt.title(f"Transaction Count and Fraud Rate vs {feature}")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return stats

    def plot_class_balance(self, target: str = 'class', labels: tuple = ("Not Fraud (0)", "Fraud (1)"),
                           figsize: tuple = (6, 6), autopct: str = '%1.1f%%') -> pd.Series:
        """Plot a pie chart showing class imbalance and return counts.

        Parameters
        - target: column name for the binary target (0/1)
        - labels: tuple of labels for (0, 1)
        - figsize: figure size
        - autopct: format for percentage labels on the pie

        Returns a Series with counts indexed by target value.
        """
        if target not in self.df.columns:
            raise ValueError(f"Column '{target}' not found in DataFrame")

        counts = self.df[target].value_counts().sort_index()
        # Ensure both classes 0 and 1 present for consistent plotting
        for cls in (0, 1):
            if cls not in counts.index:
                counts.loc[cls] = 0
        counts = counts.sort_index()

        plt.figure(figsize=figsize)
        colors = ['#66b3ff', '#ff6666']
        plt.pie(counts.values, labels=[f"{labels[i]}: {int(counts.values[i])}" for i in range(len(counts))],
                autopct=autopct, colors=colors, startangle=90, counterclock=False)
        plt.title('Class Balance (transactions)')
        plt.tight_layout()
        plt.show()

        return counts


