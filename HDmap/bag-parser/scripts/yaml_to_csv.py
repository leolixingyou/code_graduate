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

argparser.add_argument(
    '--bagname',
    default='None',
    help='bagname'
)


args = argparser.parse_args()
filename = args.filename
bagname = args.bagname

with open(f'yamls/{bagname}/{filename}.yaml') as f:

    raws = yaml.load_all(f, Loader=yaml.FullLoader)
    raws = list(raws)

    col_names = ['unix_time', 'latitude', 'longitude']

    rows = len(raws)
    cols = len(col_names)

    arr = np.zeros([rows, cols], dtype=np.float64)
    
    for i, raw in enumerate(raws):
        header = raw['header']

        #arr[i, 0] = raw['time_stamp']
        arr[i, 0] = f"{header['stamp']['secs']}.{header['stamp']['nsecs']}"
        arr[i, 1] = raw['latitude']
        arr[i, 2] = raw['longitude']
        #arr[i, 3] = raw['altitude']

        print(arr[1,:])

    pd.DataFrame(arr).to_csv(f"csvs/{bagname}/{filename}.csv", header=col_names, index=None)

print(f"Saved {rows} data in csvs/{bagname}/{filename}.csv from yamls/{bagname}/{filename}.yaml")