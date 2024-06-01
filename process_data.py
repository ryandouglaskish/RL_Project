import pandas as pd
import os


def concat_csv_files(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    all_dataframes = [pd.read_csv(file) for file in all_files]
    for i, df in enumerate(all_dataframes):
        assert df.isna().sum().sum() == 0, f"Dataframe {i} contains missing values"
    concatenated_df = pd.concat(all_dataframes, ignore_index=True)
    return concatenated_df

# Example usage
directory = 'data/raw'  # Change this to the actual directory containing your CSV files
result_df = concat_csv_files(directory)
if not os.path.exists('data/processed'):
    os.makedirs('data/processed')
result_df.to_csv('data/processed/concatenated_data.csv', index=False)
