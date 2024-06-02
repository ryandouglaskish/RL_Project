import pandas as pd
import os

if __name__ == '__main__':
    df = pd.read_csv('data/processed/standard.csv')
    print(df.shape)