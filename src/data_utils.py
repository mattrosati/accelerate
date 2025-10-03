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

    df[f"elapsed_{time}"] = (df["DateTime"] - df["DateTime"].iloc[0]).dt.total_seconds()
    df.set_index(df[f"elapsed_{time}"], inplace=True)

    return df


def find_time_elapsed(ptid, calc, path):
    df = load_label(ptid, labels_path=path)
    try:
        return df[calc][df[calc].notna()].index[0]
    except:
        print(f"No time found for {ptid}")


def build_continuous_time(f, variable_path):
    # input: file and name of series
    # return: continuous time array and values array with nans in gaps
    index = pd.DataFrame(f[variable_path].attrs["index"])
    step_size = (
        index["frequency"] * 1e6
    )  # according to sampling frequency, converts to microseconds
    segment_end_times = index["starttime"] + index["length"] * step_size
    gaps_start_times = segment_end_times[:-1].reset_index(drop=True)
    gaps_end_times = index["starttime"][1:].reset_index(drop=True)
    gaps_length = gaps_end_times - gaps_start_times

    # need a index, time in unix dataframe
    gaps_length = gaps_length / step_size[:-1].reset_index(drop=True)
    time = np.arange(0, int(gaps_length.sum().item() + f[variable_path][:].shape[0]))
    print(time.shape)

    # mask for gap or not gap
    start_end = pd.DataFrame(
        {
            "start": (gaps_start_times - index["starttime"].iloc[0].item())
            / step_size[:-1].reset_index(drop=True),
            "end": (gaps_end_times - index["starttime"].iloc[0].item())
            / step_size[:-1].reset_index(drop=True),
        }
    )
    index["zero_start"] = round(
        (index["starttime"] - index["starttime"].iloc[0].item()) / step_size
    )
    mask = np.ones(time.shape[0], dtype=bool)
    # check if time is in any gap
    for i in range(start_end.shape[0]):
        mask[
            int(start_end["start"].iloc[i].item()) : int(
                start_end["end"].iloc[i].item()
            )
        ] = False
    print(mask.sum() == f[variable_path][:].shape[0])

    # fill gaps with nans and not gaps with values
    values = np.empty(time.shape[0])
    values[:] = np.nan
    values[mask] = f[variable_path][:]
    df = pd.DataFrame(values, index=time)
    return df
