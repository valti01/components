import numpy as np
import pandas as pd

filename = '2025_12_results_copy.csv'

data_frame = pd.read_csv(f'{filename}', delimiter=',')

data_frame.to_csv(f'2025_12_results_copy.csv', sep=';', index=False)
