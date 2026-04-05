"""
Label & Data Integrity Audit Script.

Runs 6 checks on a processed dataset directory to verify label correctness,
class balance, data-label alignment, and multiclass ar_class consistency.
See --help for usage.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def load_labels(dataset_dir):
    """Load train and test labels.pkl files."""
    train_labels = pd.read_pickle(os.path.join(dataset_dir, "train", "labels.pkl"))
    test_labels = pd.read_pickle(os.path.join(dataset_dir, "test", "labels.pkl"))
    print(f"Train labels columns: {list(train_labels.columns)}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print()
    return train_labels, test_labels


# ---------------------------------------------------------------------------
# Check 1: Class Balance
# ---------------------------------------------------------------------------


def check_class_balance(train_labels, test_labels, output_dir):
    """Analyze class distribution of in? across train/test and per-patient."""
    results = {}

    for name, labels in [("train", train_labels), ("test", test_labels)]:
        total = len(labels)
        n_nan = int(labels["in?"].isna().sum())
        valid_in = labels["in?"].dropna().astype(float)
        n_true = int((valid_in == 1.0).sum())
        n_false = int((valid_in == 0.0).sum())

        results[name] = {
            "total": total,
            "in_true": n_true,
            "in_false": n_false,
            "in_nan": n_nan,
            "pct_true": round(100 * n_true / total, 2) if total > 0 else 0,
            "pct_false": round(100 * n_false / total, 2) if total > 0 else 0,
            "pct_nan": round(100 * n_nan / total, 2) if total > 0 else 0,
        }

        # ar_class distribution (multiclass: 0=below, 1=in, 2=above)
        if "ar_class" in labels.columns:
            ar = labels["ar_class"].dropna()
            ar_counts = {int(c): int((ar == c).sum()) for c in [0, 1, 2]}
            ar_nan = int(labels["ar_class"].isna().sum())
            results[name]["ar_class"] = {
                "below": ar_counts.get(0, 0),
                "in": ar_counts.get(1, 0),
                "above": ar_counts.get(2, 0),
                "nan": ar_nan,
            }
            if len(ar) > 0:
                for c in [0, 1, 2]:
                    results[name]["ar_class"][f"pct_{c}"] = round(
                        100 * ar_counts.get(c, 0) / total, 2
                    )

            # ar_class <-> in? consistency
            both_valid = labels.dropna(subset=["ar_class", "in?"])
            if len(both_valid) > 0:
                ar_in = both_valid["ar_class"] == 1
                in_true = both_valid["in?"].astype(float) == 1.0
                match = float((ar_in == in_true).mean())
                results[name]["ar_class_in_consistency"] = round(match, 4)

        # frac_out distribution
        if "frac_out" in labels.columns:
            frac_out = labels["frac_out"].dropna()
            if len(frac_out) > 0:
                results[name]["frac_out_percentiles"] = {
                    str(p): round(float(np.percentile(frac_out, p)), 4)
                    for p in [5, 25, 50, 75, 95]
                }

        # Per-patient balance
        patient_col = "base_ptid" if "base_ptid" in labels.columns else "ptid"
        valid = labels.dropna(subset=["in?"])
        if len(valid) > 0:
            patient_balance = valid.groupby(patient_col)["in?"].mean()
            all_positive = int((patient_balance == 1.0).sum())
            all_negative = int((patient_balance == 0.0).sum())
            results[name]["n_patients"] = int(patient_balance.shape[0])
            results[name]["patients_all_positive"] = all_positive
            results[name]["patients_all_negative"] = all_negative
            results[name]["patient_in_rate_mean"] = round(
                float(patient_balance.mean()), 4
            )
            results[name]["patient_in_rate_std"] = round(
                float(patient_balance.std()), 4
            )

    # Print summary
    print("=" * 60)
    print("CHECK 1: CLASS BALANCE")
    print("=" * 60)
    for name in ["train", "test"]:
        r = results[name]
        print(f"\n  {name.upper()} ({r['total']} windows):")
        print(f"    in?=True:  {r['in_true']:>6} ({r['pct_true']:.1f}%)")
        print(f"    in?=False: {r['in_false']:>6} ({r['pct_false']:.1f}%)")
        print(f"    in?=NaN:   {r['in_nan']:>6} ({r['pct_nan']:.1f}%)")
        if "n_patients" in r:
            print(f"    Patients: {r['n_patients']}")
            print(
                f"    Patients 100% positive: {r['patients_all_positive']}, "
                f"100% negative: {r['patients_all_negative']}"
            )
            print(
                f"    Patient-level in-rate: {r['patient_in_rate_mean']:.3f} "
                f"+/- {r['patient_in_rate_std']:.3f}"
            )
        if "frac_out_percentiles" in r:
            print(f"    frac_out percentiles: {r['frac_out_percentiles']}")
        if "ar_class" in r:
            ac = r["ar_class"]
            print(f"    ar_class: below={ac['below']}, in={ac['in']}, above={ac['above']}, nan={ac['nan']}")
            for c, label in [(0, "below"), (1, "in"), (2, "above")]:
                pct = ac.get(f"pct_{c}", 0)
                print(f"      {label}: {pct:.1f}%")
        if "ar_class_in_consistency" in r:
            cons = r["ar_class_in_consistency"]
            print(f"    ar_class <-> in? consistency: {cons:.4f}")
            if cons < 0.99:
                print(f"    *** WARNING: {100*(1-cons):.1f}% of ar_class labels inconsistent with in?")

    # Plot class balance per patient
    if output_dir:
        valid = train_labels.dropna(subset=["in?"])
        patient_col = "base_ptid" if "base_ptid" in valid.columns else "ptid"
        if len(valid) > 0:
            patient_balance = valid.groupby(patient_col)["in?"].mean().sort_values()
            has_frac_out = "frac_out" in train_labels.columns
            ncols = 2 if has_frac_out else 1
            fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
            if ncols == 1:
                axes = [axes]

            axes[0].barh(range(len(patient_balance)), patient_balance.values)
            axes[0].set_xlabel("Fraction in? = True")
            axes[0].set_ylabel("Patient index")
            axes[0].set_title("Per-patient class balance (train)")
            axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.5)

            if has_frac_out:
                frac_out = train_labels["frac_out"].dropna()
                axes[1].hist(frac_out, bins=50, edgecolor="black", alpha=0.7)
                axes[1].set_xlabel("frac_out")
                axes[1].set_ylabel("Count")
                axes[1].set_title("frac_out distribution (train)")
                axes[1].axvline(
                    0.46, color="red", linestyle="--", label="threshold=0.46"
                )
                axes[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "class_balance.png"), dpi=150)
            plt.close()

        # ar_class per-patient stacked bar
        if "ar_class" in train_labels.columns:
            valid_ar = train_labels.dropna(subset=["ar_class"])
            if len(valid_ar) > 0:
                patient_col = "base_ptid" if "base_ptid" in valid_ar.columns else "ptid"
                ct = valid_ar.groupby(patient_col)["ar_class"].value_counts(normalize=True).unstack(fill_value=0)
                ct = ct.reindex(columns=[0, 1, 2], fill_value=0)
                ct = ct.sort_values(1)  # sort by fraction "in"
                fig, ax = plt.subplots(figsize=(8, max(5, len(ct) * 0.2)))
                ct.plot.barh(stacked=True, ax=ax, color=["#d62728", "#2ca02c", "#ff7f0e"])
                ax.set_xlabel("Fraction")
                ax.set_ylabel("Patient")
                ax.set_title("Per-patient ar_class distribution (train)")
                ax.legend(["below", "in", "above"], loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "ar_class_balance.png"), dpi=150)
                plt.close()

    return results


# ---------------------------------------------------------------------------
# Check 2: Trivial Classifier Sanity Check
# ---------------------------------------------------------------------------


def check_trivial_classifier(train_labels, test_labels):
    """Test if simple features can predict in? — diagnostic for label integrity."""
    results = {}

    print("\n" + "=" * 60)
    print("CHECK 2: TRIVIAL CLASSIFIER SANITY CHECK")
    print("=" * 60)

    # Combine train for CV
    labels = train_labels.copy()
    has_frac_out = "frac_out" in labels.columns
    has_mapopt = "MAPopt_Yale_affected_beta" in labels.columns
    patient_col = "base_ptid" if "base_ptid" in labels.columns else "ptid"

    if not has_frac_out and not has_mapopt:
        print("  Neither frac_out nor MAPopt_Yale_affected_beta found in labels.")
        print(f"  Available columns: {list(labels.columns)}")
        return results

    # Check 2a: frac_out -> in? (should be ~1.0 AUC)
    if has_frac_out:
        valid = labels.dropna(subset=["in?", "frac_out"])
        if len(valid) >= 10:
            y = valid["in?"].astype(int).values
            groups = valid[patient_col].values
            X_frac = valid[["frac_out"]].values
            try:
                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
                aucs = []
                for train_idx, val_idx in gkf.split(X_frac, y, groups):
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_frac[train_idx], y[train_idx])
                    proba = clf.predict_proba(X_frac[val_idx])[:, 1]
                    if len(np.unique(y[val_idx])) > 1:
                        aucs.append(roc_auc_score(y[val_idx], proba))
                frac_out_auc = float(np.mean(aucs)) if aucs else float("nan")
                results["frac_out_to_in_auc"] = round(frac_out_auc, 4)
                print(f"\n  frac_out -> in? AUC: {frac_out_auc:.4f}")
                if frac_out_auc < 0.95:
                    print("  *** WARNING: frac_out -> in? AUC < 0.95!")
                    print("  *** This suggests in? is NOT consistently derived from frac_out.")
                else:
                    print("  OK: frac_out correctly determines in?")
            except Exception as e:
                print(f"  Error in frac_out check: {e}")
        else:
            print("  Not enough valid samples for frac_out check.")
    else:
        print("  frac_out column not found in labels — skipping Check 2a.")

    # Check 2b: MAPopt -> in? (expect moderate AUC)
    if has_mapopt:
        valid2 = labels.dropna(subset=["in?", "MAPopt_Yale_affected_beta"])
        if len(valid2) > 10:
            X_mapopt = valid2[["MAPopt_Yale_affected_beta"]].values
            y2 = valid2["in?"].astype(int).values
            groups2 = valid2[patient_col].values
            try:
                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups2))))
                aucs = []
                for train_idx, val_idx in gkf.split(X_mapopt, y2, groups2):
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_mapopt[train_idx], y2[train_idx])
                    proba = clf.predict_proba(X_mapopt[val_idx])[:, 1]
                    if len(np.unique(y2[val_idx])) > 1:
                        aucs.append(roc_auc_score(y2[val_idx], proba))
                mapopt_auc = float(np.mean(aucs)) if aucs else float("nan")
                results["mapopt_to_in_auc"] = round(mapopt_auc, 4)
                print(f"  MAPopt -> in? AUC: {mapopt_auc:.4f}")
            except Exception as e:
                print(f"  Error in MAPopt check: {e}")
    else:
        print("  MAPopt_Yale_affected_beta column not found — skipping Check 2b.")

    # Check 2d: frac_out -> ar_class (multiclass trivial classifier)
    if has_frac_out and "ar_class" in labels.columns:
        valid_mc = labels.dropna(subset=["ar_class", "frac_out"])
        if len(valid_mc) >= 10 and len(np.unique(valid_mc["ar_class"])) > 1:
            X_mc = valid_mc[["frac_out"]].values
            y_mc = valid_mc["ar_class"].astype(int).values
            groups_mc = valid_mc[patient_col].values
            try:
                from sklearn.metrics import balanced_accuracy_score

                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_mc))))
                baccs = []
                mc_aucs = []
                for train_idx, val_idx in gkf.split(X_mc, y_mc, groups_mc):
                    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
                    clf.fit(X_mc[train_idx], y_mc[train_idx])
                    y_pred = clf.predict(X_mc[val_idx])
                    baccs.append(balanced_accuracy_score(y_mc[val_idx], y_pred))
                    if len(np.unique(y_mc[val_idx])) > 1:
                        proba = clf.predict_proba(X_mc[val_idx])
                        try:
                            mc_aucs.append(
                                roc_auc_score(
                                    y_mc[val_idx], proba,
                                    multi_class="ovr", average="macro",
                                )
                            )
                        except ValueError:
                            pass
                mean_bacc = float(np.mean(baccs)) if baccs else float("nan")
                mean_mc_auc = float(np.mean(mc_aucs)) if mc_aucs else float("nan")
                results["frac_out_to_ar_class_bacc"] = round(mean_bacc, 4)
                results["frac_out_to_ar_class_auc"] = round(mean_mc_auc, 4)
                print(f"\n  frac_out -> ar_class balanced accuracy: {mean_bacc:.4f}")
                print(f"  frac_out -> ar_class macro AUC (OVR): {mean_mc_auc:.4f}")
                print(
                    "  (Moderate performance expected: frac_out alone can't distinguish below vs above)"
                )
            except Exception as e:
                print(f"  Error in ar_class trivial classifier check: {e}")
        else:
            print("  Not enough valid multiclass samples for ar_class check.")

    # Check 2c: Consistency check — does in? == (frac_out <= threshold)?
    if has_frac_out:
        from constants import SMOOTH_FRAC_OUT_MIN

        valid = labels.dropna(subset=["in?", "frac_out"])
        derived_in = valid["frac_out"] <= SMOOTH_FRAC_OUT_MIN
        actual_in = valid["in?"].astype(bool)
        match_rate = float((derived_in == actual_in).mean())
        results["label_threshold_match_rate"] = round(match_rate, 4)
        print(f"\n  Label consistency (in? == frac_out <= {SMOOTH_FRAC_OUT_MIN}):")
        print(f"    Match rate: {match_rate:.4f}")
        if match_rate < 0.99:
            print(
                f"  *** WARNING: {100*(1-match_rate):.1f}% of labels don't match threshold rule!"
            )
            mismatches = valid[derived_in != actual_in]
            print(f"    Mismatched rows: {len(mismatches)}")
            if len(mismatches) > 0:
                print(f"    Sample mismatches (first 5):")
                print(
                    mismatches[["frac_out", "in?", patient_col]].head().to_string(
                        index=False
                    )
                )
        else:
            print("  OK: Labels are consistent with threshold rule.")
    else:
        print("  frac_out column not found — skipping consistency check.")

    return results


# ---------------------------------------------------------------------------
# Check 3: Data-Label Alignment
# ---------------------------------------------------------------------------


def check_data_label_alignment(dataset_dir, sample_n):
    """Verify that zarr arrays and labels have matching dimensions."""
    results = {}

    print("\n" + "=" * 60)
    print("CHECK 3: DATA-LABEL ALIGNMENT")
    print("=" * 60)

    for split in ["train", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        labels = pd.read_pickle(os.path.join(split_dir, "labels.pkl"))

        # Find all zarr arrays in this split
        zarr_files = [
            f for f in os.listdir(split_dir) if f.endswith(".zarr") and "x" in f
        ]
        split_results = {"n_labels": len(labels)}

        for zf in sorted(zarr_files):
            z = zarr.open(os.path.join(split_dir, zf), "r")
            shape = z.shape
            split_results[zf] = {
                "shape": list(shape),
                "matches_labels": shape[0] == len(labels),
            }
            status = "OK" if shape[0] == len(labels) else "MISMATCH"
            print(f"  {split}/{zf}: shape={shape}, labels={len(labels)} [{status}]")

            if shape[0] != len(labels):
                print(
                    f"  *** WARNING: {zf} has {shape[0]} rows but labels has {len(labels)}!"
                )

        # Check datetime monotonicity per patient
        patient_col = "base_ptid" if "base_ptid" in labels.columns else "ptid"
        if "datetime" in labels.columns:
            non_monotonic = 0
            duplicates = 0
            patients = labels[patient_col].unique()
            check_patients = patients[: min(sample_n, len(patients))]
            for p in check_patients:
                pt_labels = labels[labels[patient_col] == p]
                dt = pt_labels["datetime"].values
                if len(dt) > 1:
                    if not np.all(dt[1:] >= dt[:-1]):
                        non_monotonic += 1
                    if len(dt) != len(np.unique(dt)):
                        duplicates += 1

            split_results["non_monotonic_patients"] = non_monotonic
            split_results["duplicate_datetime_patients"] = duplicates
            print(
                f"  {split}: datetime monotonicity issues: {non_monotonic}/{len(check_patients)}"
            )
            print(
                f"  {split}: duplicate datetime issues: {duplicates}/{len(check_patients)}"
            )

        results[split] = split_results

    return results


# ---------------------------------------------------------------------------
# Check 4: ABP Signal vs Label Direction
# ---------------------------------------------------------------------------


def _parse_channels_from_path(dataset_dir):
    """Infer channel names and order from the dataset directory name.

    Directory names follow the pattern:
        ...<params>_w_<dur>_<ch1>_<ch2>_..._<chN>/permanent
    Known channels: hr, rso2r, rso2l, spo2, abp.
    """
    from constants import FEATURES

    # Walk up from permanent/ to the config directory name
    path = os.path.normpath(dataset_dir)
    parts = path.split(os.sep)
    # Find the config dir (parent of 'permanent' or the dir itself)
    config_name = None
    for i, p in enumerate(parts):
        if p == "permanent" and i > 0:
            config_name = parts[i - 1]
            break
    if config_name is None:
        config_name = parts[-1]

    # Extract channels: split on '_' and collect known feature names in order
    tokens = config_name.split("_")
    channels = [t for t in tokens if t in FEATURES]
    return channels if channels else None


def check_abp_label_direction(dataset_dir, sample_n):
    """Cross-check that ABP window means relate sensibly to in?/frac_out."""
    results = {}

    print("\n" + "=" * 60)
    print("CHECK 4: ABP SIGNAL vs LABEL DIRECTION")
    print("=" * 60)

    split_dir = os.path.join(dataset_dir, "train")
    labels = pd.read_pickle(os.path.join(split_dir, "labels.pkl"))

    # Try to find the main x.zarr (combined) or abp_x.zarr
    zarr_candidates = ["x.zarr", "abp_x.zarr"]
    zarr_path = None
    for zc in zarr_candidates:
        candidate = os.path.join(split_dir, zc)
        if os.path.exists(candidate):
            zarr_path = candidate
            break

    if zarr_path is None:
        print("  Could not find x.zarr or abp_x.zarr in train/. Skipping.")
        return results

    z = zarr.open(zarr_path, "r")
    n_features = z.shape[1]
    print(f"  Using {zarr_path}, shape={z.shape}")

    # Determine which columns belong to ABP
    channels = _parse_channels_from_path(dataset_dir)
    abp_slice = None
    if channels and "abp" in channels:
        n_channels = len(channels)
        samples_per_channel = n_features // n_channels
        abp_idx = channels.index("abp")
        abp_start = abp_idx * samples_per_channel
        abp_end = abp_start + samples_per_channel
        abp_slice = slice(abp_start, abp_end)
        print(
            f"  Detected channels: {channels} "
            f"({samples_per_channel} samples each)"
        )
        print(f"  ABP channel: columns [{abp_start}:{abp_end}]")
    elif zarr_path.endswith("abp_x.zarr"):
        abp_slice = slice(None)
        print("  Using abp_x.zarr (single-channel)")
    else:
        print(
            f"  Could not determine ABP channel position from path. "
            f"Using all {n_features} columns (correlation may be weak)."
        )

    # Use frac_out if available, otherwise fall back to in?
    target_col = "frac_out" if "frac_out" in labels.columns else "in?"
    valid = labels.dropna(subset=[target_col]).reset_index(drop=True)
    valid_idx = labels.dropna(subset=[target_col]).index.values

    n_check = min(sample_n * 100, len(valid_idx), z.shape[0])
    if n_check < 10:
        print("  Not enough valid samples for ABP-label check.")
        return results

    # Sample windows and compute mean of ABP channel only
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(valid_idx), size=n_check, replace=False)
    zarr_idx = valid_idx[sample_idx]

    window_means = []
    for idx in zarr_idx:
        if idx < z.shape[0]:
            w = z[int(idx)]
            abp_data = w[abp_slice] if abp_slice is not None else w
            window_means.append(float(np.nanmean(abp_data)))

    window_means = np.array(window_means)
    target_vals = valid.iloc[sample_idx][target_col].astype(float).values[
        : len(window_means)
    ]

    # Correlation between ABP window mean and target
    valid_mask = ~np.isnan(window_means) & ~np.isnan(target_vals)
    if valid_mask.sum() > 10:
        corr = float(
            np.corrcoef(window_means[valid_mask], target_vals[valid_mask])[0, 1]
        )
        results[f"abp_mean_{target_col}_correlation"] = round(corr, 4)
        print(f"\n  Correlation(ABP_mean, {target_col}) = {corr:.4f}")
        if abs(corr) < 0.1:
            print(
                "  *** WARNING: Very weak correlation between ABP signal mean "
                "and label. This may indicate a data-label alignment issue, "
                "or that the signal has been normalized in a way that removes "
                "the ABP level information."
            )

        # Also report per-channel correlations for context
        if channels and len(channels) > 1:
            samples_per_channel = n_features // len(channels)
            print(f"\n  Per-channel correlations with {target_col}:")
            for ci, ch in enumerate(channels):
                ch_start = ci * samples_per_channel
                ch_end = ch_start + samples_per_channel
                ch_means = []
                for idx in zarr_idx:
                    if idx < z.shape[0]:
                        w = z[int(idx)]
                        ch_means.append(float(np.nanmean(w[ch_start:ch_end])))
                ch_means = np.array(ch_means)[: len(target_vals)]
                ch_mask = ~np.isnan(ch_means) & ~np.isnan(target_vals)
                if ch_mask.sum() > 10:
                    ch_corr = float(
                        np.corrcoef(ch_means[ch_mask], target_vals[ch_mask])[0, 1]
                    )
                    results[f"{ch}_mean_{target_col}_correlation"] = round(ch_corr, 4)
                    print(f"    {ch:>6s}: {ch_corr:+.4f}")
    else:
        print("  Not enough valid pairs to compute correlation.")

    # ar_class directional check: ABP mean should be lower for class 0, higher for class 2
    if "ar_class" in labels.columns and abp_slice is not None:
        ar_valid = valid.iloc[sample_idx[:len(window_means)]].copy()
        ar_valid["abp_mean"] = window_means[:len(ar_valid)]
        ar_valid = ar_valid.dropna(subset=["ar_class", "abp_mean"])

        if len(ar_valid) > 0 and len(ar_valid["ar_class"].unique()) > 1:
            print(f"\n  ABP mean by ar_class:")
            class_means = {}
            for c, label in [(0, "below"), (1, "in"), (2, "above")]:
                subset = ar_valid[ar_valid["ar_class"] == c]["abp_mean"]
                if len(subset) > 0:
                    m = float(subset.mean())
                    class_means[c] = m
                    print(f"    {label} (n={len(subset)}): mean ABP = {m:.4f}")
                else:
                    print(f"    {label}: no samples")

            # Check ordering: below < in < above (for normalized data, check relative ordering)
            if 0 in class_means and 1 in class_means and 2 in class_means:
                ordering_ok = class_means[0] < class_means[1] < class_means[2]
                results["ar_class_abp_ordering_correct"] = ordering_ok
                if ordering_ok:
                    print("  OK: ABP mean ordering is below < in < above")
                else:
                    print(
                        f"  *** NOTE: ABP ordering is {class_means[0]:.4f} / "
                        f"{class_means[1]:.4f} / {class_means[2]:.4f}. "
                        "Expected below < in < above. This may be normal if data is normalized."
                    )

            # Kruskal-Wallis test
            try:
                from scipy.stats import kruskal

                groups_by_class = [
                    ar_valid[ar_valid["ar_class"] == c]["abp_mean"].values
                    for c in [0, 1, 2]
                    if len(ar_valid[ar_valid["ar_class"] == c]) > 0
                ]
                if len(groups_by_class) >= 2:
                    stat, pval = kruskal(*groups_by_class)
                    results["ar_class_kruskal_stat"] = round(float(stat), 4)
                    results["ar_class_kruskal_pval"] = float(pval)
                    print(f"  Kruskal-Wallis: H={stat:.2f}, p={pval:.2e}")
                    if pval > 0.05:
                        print("  *** WARNING: ar_class groups are NOT significantly different in ABP mean")
            except ImportError:
                print("  scipy not available — skipping Kruskal-Wallis test")

    return results


# ---------------------------------------------------------------------------
# Check 5: Window Filtering Summary
# ---------------------------------------------------------------------------


def check_window_filtering(train_labels, test_labels):
    """Summarize window counts and invalids per patient."""
    results = {}

    print("\n" + "=" * 60)
    print("CHECK 5: WINDOW FILTERING SUMMARY")
    print("=" * 60)

    for name, labels in [("train", train_labels), ("test", test_labels)]:
        patient_col = "base_ptid" if "base_ptid" in labels.columns else "ptid"
        patients = labels[patient_col].unique()

        window_counts = labels.groupby(patient_col).size()
        invalid_counts = (
            labels[labels.get("invalid", pd.Series(False, index=labels.index)) == True]
            .groupby(patient_col)
            .size()
        )

        results[name] = {
            "n_patients": int(len(patients)),
            "total_windows": int(len(labels)),
            "windows_per_patient_mean": round(float(window_counts.mean()), 1),
            "windows_per_patient_median": round(float(window_counts.median()), 1),
            "windows_per_patient_min": int(window_counts.min()),
            "windows_per_patient_max": int(window_counts.max()),
        }

        if len(invalid_counts) > 0:
            results[name]["total_invalid"] = int(invalid_counts.sum())
        else:
            results[name]["total_invalid"] = 0

        print(f"\n  {name.upper()}:")
        r = results[name]
        print(f"    Patients: {r['n_patients']}")
        print(f"    Total windows: {r['total_windows']}")
        print(
            f"    Windows/patient: mean={r['windows_per_patient_mean']}, "
            f"median={r['windows_per_patient_median']}, "
            f"min={r['windows_per_patient_min']}, max={r['windows_per_patient_max']}"
        )
        print(f"    Invalid (NaN labels): {r['total_invalid']}")

        # Flag patients with very few windows
        small_patients = window_counts[window_counts < 10]
        if len(small_patients) > 0:
            print(
                f"    Patients with <10 windows: {len(small_patients)} "
                f"({list(small_patients.index[:5])}{'...' if len(small_patients) > 5 else ''})"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Audit label and data integrity for a processed dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to the permanent/ directory of a dataset config.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for saving plots and JSON report. Defaults to dataset_dir/audit/.",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=20,
        help="Number of patients to spot-check for alignment (default: 20).",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir or os.path.join(dataset_dir, "audit")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Auditing dataset: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load labels
    train_labels, test_labels = load_labels(dataset_dir)

    # Run checks
    all_results = {}
    all_results["check1_class_balance"] = check_class_balance(
        train_labels, test_labels, output_dir
    )
    all_results["check2_trivial_classifier"] = check_trivial_classifier(
        train_labels, test_labels
    )
    all_results["check3_alignment"] = check_data_label_alignment(
        dataset_dir, args.sample_n
    )
    all_results["check4_abp_direction"] = check_abp_label_direction(
        dataset_dir, args.sample_n
    )
    all_results["check5_filtering"] = check_window_filtering(
        train_labels, test_labels
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    issues = []
    c2 = all_results.get("check2_trivial_classifier", {})
    if c2.get("frac_out_to_in_auc", 1.0) < 0.95:
        issues.append(
            f"frac_out -> in? AUC is {c2['frac_out_to_in_auc']:.3f} (expected ~1.0)"
        )
    if c2.get("label_threshold_match_rate", 1.0) < 0.99:
        issues.append(
            f"Label-threshold match rate is {c2['label_threshold_match_rate']:.3f}"
        )

    c3 = all_results.get("check3_alignment", {})
    for split in ["train", "test"]:
        if split in c3:
            for k, v in c3[split].items():
                if isinstance(v, dict) and not v.get("matches_labels", True):
                    issues.append(f"{split}/{k}: shape mismatch with labels")

    # ar_class checks
    c1 = all_results.get("check1_class_balance", {})
    if "ar_class" not in c1.get("train", {}):
        issues.append("ar_class column missing from train labels")
    for split_name in ["train", "test"]:
        cons = c1.get(split_name, {}).get("ar_class_in_consistency")
        if cons is not None and cons < 0.99:
            issues.append(
                f"{split_name}: ar_class <-> in? consistency is {cons:.3f} (expected ~1.0)"
            )
    c4 = all_results.get("check4_abp_direction", {})
    if c4.get("ar_class_abp_ordering_correct") is False:
        issues.append("ABP mean ordering does not follow below < in < above")
    if c4.get("ar_class_kruskal_pval", 0) > 0.05:
        issues.append("ar_class groups not significantly different in ABP mean (Kruskal-Wallis)")

    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  No critical issues detected.")
        print("  If model performance is still low, the signal itself may not")
        print("  contain sufficient information for the classification task.")

    # Save JSON report
    report_path = os.path.join(output_dir, "audit_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
