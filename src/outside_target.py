# here I want to calculate:
# - % time outside of target across whole dataset
# - duration of time outside of target for whole dataset, with summary statistics

import os
import sys
from argparse import ArgumentParser

import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from filelock import FileLock

from data_utils import build_continuous_time, load_label
from constants import TARGETS, FEATURES



def extractor(ptid, file_path, mode):
    with h5py.File(file_path, "r") as f:
        #load labels[targets] and abp

        # build empty dataset same rows as labels


        # using .map or .apply, loop through each label timepoint

            # compute abp indexes according to modality 

            # check whether overlapping with gap
            
            # if (proportion in - proportion out) > proportion gap
                # evaluate whether to write True or False
                # attach (start, end) to use later if we want to extract actual data vectors
            # check



def main(ptid, file_path, mode):
    # extract data
    data = extractor(ptid, file_path, mode)

    # write data
    with FileLock(file_path + ".lock"):
        with h5py.File(file_path, "r+") as f:
            grp = f["{ptid}/processed"]
            grp.require_dataset("in_out", data=data["status"], dtype=data.dtype)
            grp.require_dataset("in_out_indeces", data=data[["start", "end"]], dtype=data.dtype)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract information about status within limits for each patient."
    )

    parser.add_argument("data_file", help="Path to combined dataset")
    parser.add_argument(
        "-m",
        "--mode",
        help="Specify way to calculate status in autoregulation",
        type=str,
        choices=["before", "after", "within"],
        required=True,
    )
    args = parser.parse_args()

    np.random.seed(420)

    # need to output dataframe per patient N x 1 with True or False depending on whether timepoint is in (True) or out (False)
    # N/A if window overlaps with a gap which makes 50% identification impossible

    # func_tbd needs to load dataset
    # global_f = h5py.File(args.data_file, "r+")

    # load only to get length
    with h5py.File(args.data_file, "r") as f:
        ptids = list(global_f.keys())

    # will do everything and write to file
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        func = partial(main, 
                    file_path = args.data_file,
                    mode = args.mode 
                )
        
        results = list(
            tqdm(
                ex.map(func, ptids), 
                total=ptids,
                )
        )


#     futures = [ex.submit(process_feature, feat) for feat in features]
# for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#     feature, result = fut.result()
