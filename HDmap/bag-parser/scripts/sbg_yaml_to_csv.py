import yaml

import numpy as np
import pandas as pd

import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '--filename',
    default='None',
    help='filename'
)

args = argparser.parse_args()

filename = args.filename

with open(f'yamls/{filename}.yaml') as f:

    raws = yaml.load_all(f, Loader=yaml.FullLoader)
    raws = list(raws)

    col_names = ['ros_time', 'unix_time', 'latitude', 'longitude', 'altitude']

    rows = len(raws)
    cols = len(col_names)

    arr = np.zeros([rows, cols], dtype=np.float64)
    
    for i, raw in enumerate(raws):
        header = raw['header']

        arr[i, 0] = raw['time_stamp']
        arr[i, 1] = f"{header['stamp']['secs']}.{header['stamp']['nsecs']}"
        arr[i, 2] = raw['latitude']
        arr[i, 3] = raw['longitude']
        arr[i, 4] = raw['altitude']

        print(arr[1,:])

    pd.DataFrame(arr).to_csv(f"csvs/{filename}.csv", header=col_names, index=None)

print(f"Saved {rows} data in csvs/{filename}.csv from yamls/{filename}.yaml")