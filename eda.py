import pandas as pd
import os
import matplotlib.pyplot as plt

from ansi import cyan, orange

RAW_DATA_PATH = 'data/raw'

def get_year(filename):
    return filename.split('-')[1][:4]

def plot_versus_date(df, date_column, value_column, title):
    df[value_column].plot()
    plt.title(f'{value_column} versus {date_column} - {title}')
    plt.savefig(f'viz/eda/{value_column}_versus_{date_column}_{title}.png')
    plt.close()

if __name__ == '__main__':

    if not os.path.exists('viz/eda'):
        os.makedirs('viz/eda')

    n_samples = 0
    merged_df = pd.DataFrame()

    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(RAW_DATA_PATH, file))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            year = get_year(file)
            cyan('Year: ' + year)
            print('Missing values: ', df.isna().sum().sum())
            for field in ['open','high','low','close','Volume BTC']:
                #plot_versus_date(df, 'date', field, year)
                print(f"Plotting {field} versus date")
            print()

            merged_df = pd.concat([merged_df, df])
            n_samples += len(df)

    orange('Total number of samples: ' + f"{n_samples:,}")

    # Plot on the merged DataFrame
    merged_df.sort_index(inplace=True)
    for field in ['open', 'high', 'low', 'close', 'Volume BTC']:
        plot_versus_date(merged_df, 'date', field, 'all_years')
        print(f"Plotting {field} versus date for all years")