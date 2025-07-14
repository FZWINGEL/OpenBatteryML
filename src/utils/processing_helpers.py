
import numpy as np
import pandas as pd

def resample_voltage_current(df: pd.DataFrame, num_points: int = 1024) -> pd.DataFrame:
    """
    Resamples voltage and current data to a fixed number of points using interpolation.

    Args:
        df (pd.DataFrame): DataFrame with 'voltage' and 'current' columns.
        num_points (int): The number of points to resample to.

    Returns:
        pd.DataFrame: A DataFrame with resampled 'voltage' and 'current' columns.
    """
    # Create a new, evenly spaced index for resampling
    resample_index = np.linspace(df.index.min(), df.index.max(), num_points)
    
    # Interpolate voltage and current to the new index
    df_resampled = pd.DataFrame(index=resample_index)
    df_resampled['voltage'] = np.interp(resample_index, df.index, df['voltage'])
    df_resampled['current'] = np.interp(resample_index, df.index, df['current'])
    
    return df_resampled

def calculate_rul(df: pd.DataFrame, end_of_life_soh: float) -> pd.DataFrame:
    """
    Calculates the Remaining Useful Life (RUL) for each cycle in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'soh' and 'cycle_number' columns.
        end_of_life_soh (float): The State of Health (SOH) threshold that defines the end of life.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'rul' column.
    """
    # Find the cycle number where SOH first drops below the end-of-life threshold
    end_of_life_cycle = df[df['soh'] <= end_of_life_soh]['cycle_number'].min()
    
    # If no cycle reaches the end of life, RUL is the total number of cycles minus the current cycle
    if pd.isna(end_of_life_cycle):
        df['rul'] = df['cycle_number'].max() - df['cycle_number']
    else:
        # RUL is the number of cycles remaining until the end of life
        df['rul'] = end_of_life_cycle - df['cycle_number']
        # Set RUL to 0 for cycles at or after the end of life
        df.loc[df['cycle_number'] >= end_of_life_cycle, 'rul'] = 0
        
    return df
