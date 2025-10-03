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

if __name__ == "__main__":
    parser = ArgumentParser(description="Building full dataset in input destination.")

    parser.add_argument("destination", help="Destination dir")
    parser.add_argument("raw_dir", help="Raw data dir")
    parser.add_argument("labels_dir", help="Labels dir")

    args = parser.parse_args()

    # make destination h5py file
    global_f = h5py.File(os.path.join(args.destination, "all_data.hdf5"), "w")
    global_f.create_group("raw")
    global_f.create_group("labels")
    global_f.create_group("time")

    # create unique set of overlapping ptids in labels and raw_data
    unique_ptid = set()
    for f in os.listdir(args.raw_dir):
        p = os.path.basename(f).split(".")[0]
        labels_name = p + "_updated.csv"
        if os.path.isfile(os.path.join(args.labels_dir, labels_name)):
            unique_ptid.add(p)

    # build external links to raw_data files and groups for labels
    for ptid in unique_ptid:
        raw_f = os.path.join(args.raw_dir, ptid + ".icmh5")
        global_f["raw"][ptid] = h5py.ExternalLink(raw_f, "/")

        # now for labels
        global_f["labels"].create_group(ptid)
        df = load_label(ptid, args.labels_dir)
        print(df)


    # for each labels dataset, process and add each column as dataset in labels group


    # continuous time and length comparisons


    # remove patients without labels and raw_data


