import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ANSI escape codes for cyan
CYAN = "\033[96m"
RESET = "\033[0m"

RAW_DATA_PATH = 'data/raw'

def get_year(filename):
    return filename.split('-')[1][:4]


def plot_versus_date(df, date_column, value_column, year):
    df[value_column].plot()
    plt.title(f'{value_column} versus {date_column} in {year}')
    plt.savefig(f'viz/eda/{value_column}_versus_{date_column}_in_{year}.png')
    plt.close()

if __name__ == '__main__':

    if not os.path.exists('viz/eda'):
        os.makedirs('viz/eda')

    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(RAW_DATA_PATH, file))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            year = get_year(file)
            print(CYAN + 'Year: ' + year + RESET)
            print('Missing values: ', df.isna().sum().sum())
            for field in ['open','high','low','close','Volume BTC']:
                plot_versus_date(df, 'date', field, year)
                print(f"Plotting {field} versus date")
            print()
            