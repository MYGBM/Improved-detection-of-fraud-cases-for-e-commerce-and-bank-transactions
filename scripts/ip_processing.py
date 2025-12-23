import pandas as pd

def transform_ip_to_country(fraud_df: pd.DataFrame, ip_mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges fraud data with IP-to-country mapping data.
    
    Parameters:
    - fraud_df: DataFrame containing 'ip_address' column.
    - ip_mapping_df: DataFrame containing 'lower_bound_ip_address', 'upper_bound_ip_address', 'country'.
    
    Returns:
    - DataFrame with an added 'country' column.
    """
    # Avoid modifying original dataframes
    fraud_df = fraud_df.copy()
    ip_mapping_df = ip_mapping_df.copy()
    
    # Ensure IP columns are integers
    # fraud_df 'ip_address' might be float or string, convert to int64
    # Handle potential NaNs if any (though EDA showed none)
    fraud_df['ip_address'] = fraud_df['ip_address'].fillna(0).astype('int64')
    
    ip_mapping_df['lower_bound_ip_address'] = ip_mapping_df['lower_bound_ip_address'].astype('int64')
    ip_mapping_df['upper_bound_ip_address'] = ip_mapping_df['upper_bound_ip_address'].astype('int64')
    
    # Sort for merge_asof
    fraud_df = fraud_df.sort_values('ip_address')
    ip_mapping_df = ip_mapping_df.sort_values('lower_bound_ip_address')
    
    # Use merge_asof to find the lower_bound that is <= ip_address
    # direction='backward' finds the last row in right df where 'on' value is <= left 'on' value
    merged_df = pd.merge_asof(
        fraud_df,
        ip_mapping_df,
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Filter matches: The IP must also be <= upper_bound_ip_address
    # If ip_address > upper_bound_ip_address, it means the IP falls in a gap between ranges
    mask = merged_df['ip_address'] <= merged_df['upper_bound_ip_address']
    
    # Keep country only where mask is True
    merged_df.loc[~mask, 'country'] = "Unknown"
    
    # Drop the helper columns from ip_mapping_df
    merged_df = merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
    
    
    return merged_df
