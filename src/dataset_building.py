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

# TODO: make a wrapper to account for the possibility that you could get "parts/raw" instead of just "raw"
# needs to be both a get_scalar and a get_df/get_group method


def continuous_time_process(hdf_obj, group):
    processed = hdf_obj["processed"]

    for i in hdf_obj[f"raw/{group}"].keys():

        cont = build_continuous_time(hdf_obj, f"raw/{group}/{i}")
        arr = cont.reset_index().to_numpy()
        processed.create_dataset(i, data=arr, dtype=arr.dtype)

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

    # check all directories exist
    try:
        os.makedirs(args.destination, exist_ok=False)
        print("Made destination directory because it did not exist.\n")
    except:
        pass

    # make destination h5py file
    global_f = h5py.File(os.path.join(args.destination, "all_data.hdf5"), "w")

    # count raw data files
    raw_files = [
        f.split(".")[0] for f in os.listdir(args.raw_dir) if f.endswith(".icmh5")
    ]
    print("Number of raw data files:", len(raw_files))

    # count patients with multiple files
    multiple_files = {}
    for pt in raw_files:
        if "_" in pt:
            mult = pt.split("_")[0]
            if mult in multiple_files.keys():
                multiple_files[mult] += 1
            else:
                multiple_files[mult] = 0
    multiple_files = {p: i for p, i in multiple_files.items() if i > 0}
    print(
        f"Patients with multiple files (n = {len(multiple_files.keys())}) with {np.array(list(multiple_files.values())).sum()} extra files:",
        list(multiple_files.keys()),
    )
    print(
        f"Number of unique patients with raw data files:",
        len(raw_files) - np.array(list(multiple_files.values())).sum(),
    )

    # create unique set of overlapping ptids in labels and raw_data
    label_files = [
        f.split("_")[0] for f in os.listdir(args.labels_dir) if f.endswith(".csv")
    ]
    raw_files_ids = [f.split("_")[0] if "_" in f else f for f in raw_files]
    unique_ptid = set(label_files).intersection(set(raw_files_ids))

    print(
        f"- Patients without label files n = {len(set(raw_files_ids).difference(set(label_files)))}:",
        list(set(raw_files_ids).difference(set(label_files))),
    )
    print("Number of Patients with label and raw data:", len(unique_ptid))

    # build external links to raw_data files and groups for labels
    print("\nBuilding global dataset...")
    no_targets = []
    for ptid in tqdm(list(unique_ptid)[:]):

        # check if this patient has multiple files, include a "part_n" group if so
        raw_f = os.path.join("../raw_data/", ptid + ".icmh5")
        pt_group = global_f.require_group(ptid)
        if ptid in multiple_files:
            mult_file_ids = [f for f in raw_files if f.startswith(ptid)]
            print(mult_file_ids)
            for i, n in enumerate(mult_file_ids):
                global_f[ptid].require_group("part_" + str(i))
                global_f[ptid]["part_" + str(i)]["raw"] = h5py.ExternalLink(raw_f, "/")
        else:
            global_f[ptid]["raw"] = h5py.ExternalLink(raw_f, "/")

        # now for labels
        df = load_label(ptid, args.labels_dir)

        # make one dataset for labels with custom dtype
        nan_value = (lambda x: float(x[ptid + "/raw"].attrs["invalidValue"][0]))(
            global_f
        )
        df = df.fillna(value=nan_value)
        arr = df.to_records(index=False)
        pt_group.create_dataset("labels", data=arr, dtype=arr.dtype)
        global_f.attrs["invalid_val"] = nan_value

        # remove if I have no targets
        df = pd.DataFrame(global_f[ptid + "/labels"][...])[TARGETS].replace(
            global_f.attrs["invalid_val"], np.nan
        )
        all_null = pd.isnull(df).all(axis=0)
        if all_null.sum() > 0:
            del global_f[ptid]
            no_targets.append(ptid)
            continue

        # prep a group for processed data
        processed = pt_group.require_group("processed")

        # some files have broken numeric data, I need a label that tells me that is the case
        try:
            test = pt_group[f"raw/numerics/hr"][:]
            pt_group["processed"].attrs["broken_numeric"] = False
            continue
        except:
            pt_group["processed"].attrs["broken_numeric"] = True

        # i don't think i actually need this
        # # create continuous time array for all raw data
        # processed = pt_group.require_group("processed")
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
    print("Successful build of global dataset.\n")

    print(
        f"Patients without calculated target data (n = {len(no_targets)}):", no_targets
    )
    print("Included Number of Patients:", len(global_f.keys()))
    broken_numerics = [
        i for i in global_f if global_f[i]["processed"].attrs["broken_numeric"]
    ]
    print(
        f"- Subset with broken numeric data (n = {len(broken_numerics)}):",
        broken_numerics,
    )
