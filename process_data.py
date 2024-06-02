import pandas as pd
import os
import numpy as np

from ansi import green, cyan, orange, red


def concat_csv_files(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    all_dataframes = [pd.read_csv(file) for file in all_files]
    for i, df in enumerate(all_dataframes):
        assert df.isna().sum().sum() == 0, f"Dataframe {i} contains missing values"
        df.drop(columns=['unix','symbol','Volume USD'], inplace=True)

    concatenated_df = pd.concat(all_dataframes, ignore_index=True)
    concatenated_df['date'] = pd.to_datetime(concatenated_df['date'])
    concatenated_df.set_index('date', inplace=True)
    concatenated_df.sort_index(inplace=True)
    return concatenated_df


def compute_delta(df):
    delta_df = df.copy()
    for field in ['open', 'high', 'low', 'close', 'Volume BTC']:
        delta_df[f'delta_{field}'] = delta_df[field].diff()
    delta_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # chop off first row
    delta_df = delta_df.iloc[1:]

    nans = delta_df.isna().sum().sum()
    if nans > 0:
        red(f"Warning: {nans} NaNs found in delta dataframe")
    return delta_df

def compute_log_delta(df):
    """
    Compute the log delta of the dataframe.

    Note: Volume BTC is not log transformed.
    
    """
    log_delta_df = df.copy()
    for field in ['open', 'high', 'low', 'close']:
        log_delta_df[f'log_delta_{field}'] = np.log(log_delta_df[field]).diff()
    # chop off first row
    log_delta_df = log_delta_df.iloc[1:]
    log_delta_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nans = log_delta_df.isna().sum().sum()
    if nans > 0:
        red(f"Warning: {nans} NaNs found in log delta dataframe")
    return log_delta_df

if __name__ == '__main__':

    directory = 'data/raw'

    # Standard data
    cyan("Processing standard data")
    result_df = concat_csv_files(directory)
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    result_df.to_csv('data/processed/standard.csv', index=True)


    # Delta data
    cyan("Processing delta data")
    result_df = concat_csv_files(directory)
    delta_df = compute_delta(result_df)
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    result_df.to_csv('data/processed/delta.csv', index=True)


    # Log delta data
    cyan("Processing log delta data")
    result_df = concat_csv_files(directory)
    log_delta_df = compute_log_delta(result_df)
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    result_df.to_csv('data/processed/log_delta.csv', index=True)
    green("Done")