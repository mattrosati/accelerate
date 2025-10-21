import os
import sys
from argparse import ArgumentParser

import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from data_utils import build_continuous_time, load_label, printname
from constants import TARGETS


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

    tmp = multiple_files.copy()
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

    # re-add in pt ids with parts
    for mult_ptid in multiple_files.keys():
        unique_ptid.discard(mult_ptid)
        unique_ptid = unique_ptid.union(
            {f"{m}_{i + 1}" for m, j in multiple_files.items() for i in range(j + 1)}
        )

    # build external links to raw_data files and groups for labels
    print("\nBuilding global dataset...")
    no_targets = []
    broken_abp = []
    for ptid in tqdm(list(unique_ptid)[:]):
        if tmp.get(ptid) == 0:
            assert len([f for f in raw_files if f.startswith(ptid + "_")]) == 1
            ptid = [f for f in raw_files if f.startswith(ptid + "_")][0]

        pt_group = global_f.require_group(ptid)
        raw_f = os.path.join("../raw_data/", ptid + ".icmh5")
        global_f[ptid]["raw"] = h5py.ExternalLink(raw_f, "/")

        # some files do not have abp, I need to remove
        try:
            assert pt_group[f"raw/waves/abp"][0] is not None
        except:
            broken_abp.append(ptid)
            del global_f[ptid]
            continue

        # now for labels
        strip_ptid = ptid.split("_")[0]
        df = load_label(strip_ptid, args.labels_dir, time="us")

        # remove patient if no labels
        all_null = pd.isnull(df).all(axis=0)
        if all_null.sum() > 0:
            del global_f[ptid]
            no_targets.append(ptid)
            continue

        # check that labels actually overlap with raw recording time when recording has multiple parts
        # delete if not
        if "_" in ptid:
            raw_start = int(global_f[ptid + "/raw"].attrs["dataStartTimeUnix"][0])
            raw_end = int(global_f[ptid + "/raw"].attrs["dataEndTimeUnix"][0])
            if (raw_end < df["DateTime"].iloc[0] / 1e6) or (
                raw_start > df["DateTime"].iloc[-1] / 1e6
            ):
                del global_f[ptid]
                no_targets.append(ptid)
                continue

        # make one dataset for labels with custom dtype
        nan_value = float(global_f[ptid + "/raw"].attrs["invalidValue"][0])
        df = df.fillna(value=nan_value)
        arr = df.to_records(index=False)
        pt_group.create_dataset("labels", data=arr, dtype=arr.dtype)
        global_f.attrs["invalid_val"] = nan_value

        # prep a group for processed data
        processed = pt_group.require_group("processed")

        # some files have broken numeric data, I need a label that tells me that is the case
        try:
            assert len(pt_group[f"raw/numerics/hr"][...]) > 0
            pt_group["processed"].attrs["broken_numeric"] = False
        except:
            pt_group["processed"].attrs["broken_numeric"] = True

    print("Successful build of global dataset.\n")

    print(
        f"Patients without calculated target data (n = {len(no_targets)}):", no_targets
    )
    print(
        f"Files with absent abp data (n = {len(broken_abp)}):",
        broken_abp,
    )
    print(
        f"Total: n = { len(set([f.split('_')[0] for f in global_f.keys()])) } patients over n = { len(global_f.keys()) } files."
    )

    broken_numerics = [
        i for i in global_f if global_f[i]["processed"].attrs["broken_numeric"]
    ]
    print(
        f"- Subset with broken numeric data (n = {len(broken_numerics)}):",
        broken_numerics,
    )
    
    # add list of ptids that have good quality data as dataset in root
    not_skipped = set(global_f.keys())
    label_data_not_overlapping_raw = set(['1043', '597_1'])
    ptid_complete_list = not_skipped.difference(label_data_not_overlapping_raw, set(broken_numerics))
    
    print(f"Totally healthy files:", len(ptid_complete_list))

    global_f.create_dataset("healthy_ptids", data=list(ptid_complete_list), dtype=h5py.string_dtype())
