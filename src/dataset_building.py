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

def continuous_time_process(hdf_obj, group):
    processed = hdf_obj["processed"]

    for i in hdf_obj[f"raw/{group}"].keys():
        cont = build_continuous_time(hdf_obj, f"raw/{group}/{i}")
        arr = cont.reset_index().to_numpy()
        processed.create_dataset(i, data = arr, dtype=arr.dtype)
        attrs = pd.DataFrame(hdf_obj[f"raw/{group}/{i}"].attrs["index"])
        processed[i].attrs["index"] = hdf_obj[f"raw/{group}/{i}"].attrs["index"]
        # print(processed[i].attrs["start"], processed[i].attrs["end"])
        # print(arr.shape)


    return processed

if __name__ == "__main__":
    parser = ArgumentParser(description="Building full dataset in input destination.")

    parser.add_argument("destination", help="Destination dir")
    parser.add_argument("raw_dir", help="Raw data dir")
    parser.add_argument("labels_dir", help="Labels dir")

    args = parser.parse_args()

    np.random.seed(420)

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
    for ptid in tqdm(list(unique_ptid)[:]):
        print(ptid)
        pt_group = global_f.create_group(ptid)
        raw_f = os.path.join("./raw_data/", ptid + ".icmh5")
        global_f[ptid]["raw"] = h5py.ExternalLink(raw_f, "/")

        # now for labels
        df = load_label(ptid, args.labels_dir)

        # make one dataset for labels with custom dtype
        nan_value = float(global_f[ptid + "/raw"].attrs["invalidValue"][0])
        df = df.fillna(value=nan_value)
        arr = df.to_records(index=False)
        pt_group.create_dataset("labels", data=arr, dtype=arr.dtype)
        global_f.attrs["invalid_val"] = nan_value
        
        # remove if I have no targets
        df = pd.DataFrame(global_f[ptid + "/labels"][...])[TARGETS].replace(global_f.attrs["invalid_val"], np.nan)
        all_null = pd.isnull(df).all(axis=0)
        if all_null.sum() > 0:
            print(ptid + " has no targets")
            del pt_group
            continue
        else:
            counter += 1 # TBD

        # i don't think i actually need this
        # # create continuous time array for all raw data
        # processed = pt_group.create_group("processed")
        # try:
        #     continuous_time_process(pt_group, "waves")
        #     continuous_time_process(pt_group, "numerics")
        # except:
        #     print(ptid)

        #     # summarize numerics and waveforms
        #     def summarize_series(name, obj, invalid_value=-99999):
        #         print(f"Dataset: {name}")
        #         df = pd.DataFrame(obj[:])
        #         df.replace(invalid_value, np.nan, inplace=True)
        #         print(f"Number of missing values: {df.isna().sum().sum()}")
        #         print(df.describe())
        #         print("\n")
        #         return

        #     print(f"Summarizing statistics for numerics and waveforms for patient {ptid}:")
        #     pt_group["raw/numerics"].visititems(summarize_series)
        #     f["raw/waves"].visititems(summarize_series)

        
        


        # compare continuous time arrays with each other and with labels
        # create a continuous time array for labels with missing times filled as simple NAs


        # if ptid == "1002":
        #     print(list(global_f.attrs.items()))


    # continuous time and length comparisons



