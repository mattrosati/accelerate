import os
import sys
from argparse import ArgumentParser

import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from data_utils import build_continuous_time, load_label
from constants import TARGETS

if __name__ == "__main__":
    parser = ArgumentParser(description="Building full dataset in input destination.")

    parser.add_argument("destination", help="Destination dir")
    parser.add_argument("raw_dir", help="Raw data dir")
    parser.add_argument("labels_dir", help="Labels dir")

    args = parser.parse_args()

    # make destination h5py file
    global_f = h5py.File(os.path.join(args.destination, "all_data.hdf5"), "w")

    # create unique set of overlapping ptids in labels and raw_data
    unique_ptid = set()
    for f in os.listdir(args.raw_dir):
        p = os.path.basename(f).split(".")[0]
        labels_name = p + "_updated.csv"
        if os.path.isfile(os.path.join(args.labels_dir, labels_name)):
            unique_ptid.add(p)

    # build external links to raw_data files and groups for labels
    counter = 0 # TBD
    for ptid in unique_ptid:
        pt_group = global_f.create_group(ptid)
        raw_f = os.path.join("./raw_data/", ptid + ".icmh5")
        global_f[ptid]["raw"] = h5py.ExternalLink(raw_f, "/")

        # now for labels
        df = load_label(ptid, args.labels_dir)

        # make one dataset for labels with custom dtype
        nan_value = float(global_f[ptid + "/raw"].attrs["invalidValue"][0])
        dt = np.dtype([(col, df[col].to_numpy().dtype) for col in df.columns])
        df = df.fillna(value=nan_value)
        arr = df.to_records(index=False)
        global_f[ptid].create_dataset("labels", data=arr, dtype=arr.dtype)
        global_f.attrs["invalid_val"] = nan_value
        
        # remove if I have no targets
        df = pd.DataFrame(global_f[ptid + "/labels"][...])[TARGETS].replace(global_f.attrs["invalid_val"], np.nan)
        all_null = pd.isnull(df).all(axis=0)
        if all_null.sum() > 0:
            print(ptid + " has no targets")
            del global_f[ptid]
        else:
            counter += 1 # TBD

        



        # if ptid == "1002":
        #     print(list(global_f.attrs.items()))

    print(counter)


    # continuous time and length comparisons



