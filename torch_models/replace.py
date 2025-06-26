import numpy as np
import pandas as pd

filename = 'bigmeasurement_results_hat.csv'

data_frame = pd.read_csv(f'{filename}', delimiter=',')

data_frame.to_csv(f'labelcsv/bigmeasurements_results_hat.csv', sep=';', index=False)
