import os
import sys

import h5py
import numpy as np
import pandas as pd

# pending useful sklearn imports


def printname(name):
    print(name)


def summarize_obj(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    if len(list(obj.attrs.keys())) != 0:
        print(f"Attributes:")
        for k in obj.attrs.keys():
            if k == "index" or k == "quality":
                print(f"{k}:")
                print(pd.DataFrame(obj.attrs[k]))
            else:
                print(f"{k}: {obj.attrs[k]}")


def h5py_summarize(file_path):
    print(f"Summarizing file {file_path}:")
    with h5py.File(file_path, "r") as f:
        print(f"Global attributes:")
        for k in f.attrs.keys():
            print(f"{k}: {f.attrs[k]}")
        print(f"\nContents:")
        f.visititems(summarize_obj)
    return


def load_label(patient_id, labels_path, time="seconds"):
    pt_file = f"{patient_id}_updated.csv"
    label_file = os.path.join(labels_path, pt_file)
    df = pd.read_csv(label_file)

    # convert time to datetime
    df["DateTime"] = pd.to_datetime(df["DateTime"], unit="D", origin="1899-12-30")
    # df.set_index("DateTime", inplace=True)

    # add column for elapsed time from start in seconds
    divisor = 1 if time == "seconds" else 60 if time == "minutes" else 3600

    # converting datetime to microseconds unix time for interoperability with raw data
    df["DateTime"] = (df["DateTime"].astype("int64") // 1000).round(-4)

    return df


def find_time_elapsed(ptid, calc, path, time="seconds"):
    df = load_label(ptid, labels_path=path)

    df[f"elapsed_{time}"] = ((df["DateTime"] - df["DateTime"].iloc[0]) // 1e6).astype("int32")
    df.set_index(df[f"elapsed_{time}"], inplace=True)

    try:
        return df[calc][df[calc].notna()].index[0]
    except:
        print(f"No time found for {ptid}")


def build_continuous_time(f, variable_path):
    # input: file and name of series
    # return: continuous time array and values array with nans in gaps
    index = pd.DataFrame(f[variable_path].attrs["index"])
    freq = index["frequency"].to_numpy()
    length = index["length"].to_numpy()
    starttime = index["starttime"].to_numpy()
    
    # print(index)
    time_blocks = []
    # print(length.sum())
    # print(f[variable_path][...].shape)
    for i in range(index.shape[0]):
        # split time interval into a grid that is based on the frequency
        grid_step = (1/freq[i]) * 1e6 # how many microseconds per grid step per segment

        # segments data
        seg_length = length[i]
        block_segment = np.empty((seg_length, 2))
        # print("segment:", block_segment.shape)

        if i > 0:
            seg_start = starttime[i] - starttime[0]
            block_segment[:, 0] = np.arange(seg_start, seg_length * grid_step + seg_start, grid_step)
            first_token = length[:i].sum().item()
            block_segment[:, 1] = f[variable_path][first_token:first_token + length[i]]
        else:
            block_segment[:, 0] = np.arange(0, seg_length * grid_step, grid_step)
            block_segment[:, 1] = f[variable_path][:length[i]]

        time_blocks.append(block_segment)

        # gaps data
        if i != index.shape[0] - 1:
            gap_length = int((starttime[i+1] - (starttime[i] + seg_length * grid_step)) // grid_step)
            if gap_length < 0:
                print("negative gap length. This is a problem.")
            else:
                block_segment = np.empty((gap_length, 2))
                # print("gap:", block_segment.shape)

        
                gap_start = length[i] * grid_step + starttime[i] - starttime[0]
                block_segment[:, 0] = np.arange(gap_start, grid_step * gap_length + gap_start, grid_step)
                block_segment[:, 1] = np.nan

                time_blocks.append(block_segment)


    time = np.concatenate(time_blocks, axis = 0)
    del time_blocks

    df = pd.DataFrame(time[:, 1], index=time[:, 0])

    return df
