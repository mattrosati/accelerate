"""Test suite for data pipeline correctness.

Targets accuracy-critical transformations: label computation, quality merging,
window extraction, NaN handling, normalization, and reshape integrity.
All tests use synthetic in-memory data -- no real patient data needed.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from process_utils import (
    merge_quality_intervals,
    finalize_segments,
    extract_proportions_count,
    extract_proportions_mean,
    extract_proportions_smooth,
    filter_na,
    impute,
    stride_filter,
    get_window,
    robust_floor,
    robust_ceil,
)
from deep.data import reshape_flat_windows, make_grouped_split
import dask.array as da
from data_extract import _bad_frac
from constants import ABP_PHYSIO_LO, ABP_PHYSIO_HI, ABP_MAX_BAD_FRAC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window(w, overlap_len=0, total_length=None):
    """Build a window dict matching the format used by extract_proportions_*."""
    w = np.asarray(w, dtype=float)
    if total_length is None:
        total_length = np.int64(len(w))
    else:
        total_length = np.int64(total_length)
    return {"w": w, "overlap_len": overlap_len, "total_length": total_length}


def _make_labels_df(**overrides):
    """Build a single-row labels DataFrame for extract_proportions helpers."""
    row = {
        "LLA_Yale_affected_beta": 60.0,
        "ULA_Yale_affected_beta": 80.0,
        "MAPopt_Yale_affected_beta": 70.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ===========================================================================
# Group 1: merge_quality_intervals
# ===========================================================================


class TestMergeQualityIntervals:
    """Sweep-line algorithm that removes bad-quality segments."""

    @staticmethod
    def _valid(rows):
        return pd.DataFrame(
            rows, columns=["starttime", "endtime", "frequency", "startidx", "length"]
        )

    @staticmethod
    def _bad(rows):
        return pd.DataFrame(rows, columns=["starttime", "endtime", "value"])

    def test_no_bad_intervals(self):
        valid = self._valid([(0, 100_000_000, 1.0, 0, 100)])
        bad = self._bad([])
        result = merge_quality_intervals(valid, bad)
        assert len(result) == 1
        assert result["starttime"].iloc[0] == 0
        assert result["endtime"].iloc[0] == 100_000_000

    def test_bad_covers_entire_segment(self):
        # Bad fully covers segment -> no usable data (or only zero-length pieces)
        valid = self._valid([(0, 100_000_000, 1.0, 0, 100)])
        bad = self._bad([(0, 100_000_000, 1)])
        result = merge_quality_intervals(valid, bad)
        # May produce zero-length pieces at boundaries; total usable duration should be ~0
        if not result.empty:
            durations = result["endtime"] - result["starttime"]
            assert (durations <= 0).all()

    def test_bad_splits_segment(self):
        # Segment from 0-100s, bad from 40-60s -> two pieces: [0,40) and [60,100)
        valid = self._valid([(0, 100_000_000, 1.0, 0, 100)])
        bad = self._bad([(40_000_000, 60_000_000, 1)])
        result = merge_quality_intervals(valid, bad)
        assert len(result) == 2
        assert result["starttime"].iloc[0] == 0
        assert result["endtime"].iloc[0] == 40_000_000
        assert result["starttime"].iloc[1] == 60_000_000
        assert result["endtime"].iloc[1] == 100_000_000

    def test_bad_at_start(self):
        valid = self._valid([(0, 100_000_000, 1.0, 0, 100)])
        bad = self._bad([(0, 20_000_000, 1)])
        result = merge_quality_intervals(valid, bad)
        # Filter out zero-length pieces that can occur at boundaries
        result = result[result["endtime"] > result["starttime"]].reset_index(drop=True)
        assert len(result) == 1
        assert result["starttime"].iloc[0] == 20_000_000
        assert result["endtime"].iloc[0] == 100_000_000

    def test_bad_at_end(self):
        valid = self._valid([(0, 100_000_000, 1.0, 0, 100)])
        bad = self._bad([(80_000_000, 100_000_000, 1)])
        result = merge_quality_intervals(valid, bad)
        assert len(result) == 1
        assert result["starttime"].iloc[0] == 0
        assert result["endtime"].iloc[0] == 80_000_000

    def test_overlapping_bad(self):
        # Two overlapping bad intervals [30-50] and [40-70] -> merged bad [30-70]
        valid = self._valid([(0, 100_000_000, 1.0, 0, 100)])
        bad = self._bad(
            [
                (30_000_000, 50_000_000, 1),
                (40_000_000, 70_000_000, 1),
            ]
        )
        result = merge_quality_intervals(valid, bad)
        assert len(result) == 2
        assert result["endtime"].iloc[0] == 30_000_000
        assert result["starttime"].iloc[1] == 70_000_000

    def test_preserves_frequency(self):
        valid = self._valid([(0, 100_000_000, 250.0, 0, 25000)])
        bad = self._bad([(40_000_000, 60_000_000, 1)])
        result = merge_quality_intervals(valid, bad)
        assert (result["frequency"] == 250.0).all()

    def test_startidx_recomputation(self):
        # Segment starts at sample 1000, freq=1 Hz, bad from 40-60s
        # Second piece starts at t=60s -> offset = 60 samples -> startidx = 1060
        valid = self._valid([(0, 100_000_000, 1.0, 1000, 100)])
        bad = self._bad([(40_000_000, 60_000_000, 1)])
        result = merge_quality_intervals(valid, bad)
        assert result["startidx"].iloc[0] == 1000  # first piece starts at original
        expected_offset = int(robust_ceil(60.0 * 1.0))
        assert result["startidx"].iloc[1] == 1000 + expected_offset


# ===========================================================================
# Group 2: extract_proportions_count
# ===========================================================================


class TestExtractProportionsCount:
    """Label computation via counting -- highest accuracy impact."""

    def test_all_inside(self):
        # All 100 values between LLA=60 and ULA=80
        w = _make_window(np.full(100, 70.0))
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        assert result["in?"][0] == 1.0
        assert result["frac_out"][0] == pytest.approx(0.0)

    def test_all_outside(self):
        """All-outside window should be labeled 0.0.

        With abs() margin check: abs(0 - 1) = 1.0 >= 0 (na + gap) → label assigned.
        proportion_in(0) > proportion_out(1) → False → 0.0.
        """
        w = _make_window(np.full(100, 50.0))
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        assert result["in?"][0] == 0.0
        assert result["frac_out"][0] == pytest.approx(1.0)

    def test_na_limits_produce_nan(self):
        w = _make_window(np.full(100, 70.0))
        labels = _make_labels_df(LLA_Yale_affected_beta=np.nan)
        result = extract_proportions_count([w], labels)
        assert np.isnan(result["in?"][0])
        assert np.isnan(result["frac_out"][0])

    def test_ambiguous_produces_nan(self):
        # 52 inside, 48 outside, with 10% gap overlap
        vals = np.concatenate([np.full(52, 70.0), np.full(48, 50.0)])
        w = _make_window(vals, overlap_len=10, total_length=100)
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        # proportion_in = 52/100, proportion_out = 1 - 0.52 = 0.48
        # margin = 0.52 - 0.48 = 0.04, threshold = 0 + 10/100 = 0.10
        # 0.04 < 0.10 -> NaN
        assert np.isnan(result["in?"][0])

    def test_proportion_arithmetic_with_nans(self):
        """Verify proportion computation with NaN values.

        NaNs are dropped before computing abp_status (line 314), so
        proportion_in = (truly inside) / total_length. Then
        proportion_out = 1 - proportion_in is a pessimistic upper bound
        that lumps NaN and gap with truly-outside. The margin check on
        line 325 accounts for this by subtracting proportion_na + proportion_gap.

        With 70 inside, 20 outside, 10 NaN in 100-length window:
        - proportion_in  = 70 / 100 = 0.70
        - proportion_out = 1 - 0.70 = 0.30  (upper bound: includes NaN)
        - proportion_na  = 10 / 100 = 0.10
        - margin = 0.70 - 0.30 = 0.40 >= 0.10 -> label assigned
        """
        inside = np.full(70, 70.0)
        outside = np.full(20, 50.0)
        nans = np.full(10, np.nan)
        vals = np.concatenate([inside, outside, nans])
        w = _make_window(vals, overlap_len=0, total_length=100)
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)

        assert result["frac_out"][0] == pytest.approx(0.30)
        # margin = 0.70 - 0.30 = 0.40 >= na(0.10) + gap(0) -> label assigned, in? = True
        assert result["in?"][0] == 1.0

    def test_gap_overlap_makes_ambiguous(self):
        # All inside, but huge gap overlap -> margin check fails
        w = _make_window(np.full(100, 70.0), overlap_len=90, total_length=100)
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        # proportion_in = 1.0, proportion_out = 0.0, gap = 0.9
        # margin = 1.0 - 0.0 = 1.0, threshold = 0 + 0.9 = 0.9
        # 1.0 >= 0.9 -> NOT NaN -> in? = True
        assert result["in?"][0] == 1.0

    def test_large_gap_with_mixed_values(self):
        # 60 inside, 40 outside, gap overlap = 40
        vals = np.concatenate([np.full(60, 70.0), np.full(40, 50.0)])
        w = _make_window(vals, overlap_len=40, total_length=100)
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        # proportion_in = 60/100 = 0.60, proportion_out = 0.40
        # margin = 0.20, threshold = 0 + 0.40 = 0.40
        # 0.20 < 0.40 -> NaN
        assert np.isnan(result["in?"][0])


# ===========================================================================
# Group 3: extract_proportions_mean
# ===========================================================================


class TestExtractProportionsMean:

    def test_inside(self):
        w = _make_window(np.full(100, 70.0))
        labels = _make_labels_df()
        result = extract_proportions_mean([w], labels)
        assert result["in?"][0] == 1.0
        assert result["frac_out"][0] == 0.0

    def test_outside_low(self):
        w = _make_window(np.full(100, 50.0))
        labels = _make_labels_df()
        result = extract_proportions_mean([w], labels)
        assert result["in?"][0] == 0.0
        assert result["frac_out"][0] == 1.0

    def test_all_nan_window(self):
        w = _make_window(np.full(100, np.nan))
        labels = _make_labels_df()
        result = extract_proportions_mean([w], labels)
        assert np.isnan(result["in?"][0])

    def test_frac_out_binary(self):
        # Mean mode always produces frac_out in {0.0, 1.0}
        for val in [50.0, 60.0, 70.0, 80.0, 90.0]:
            w = _make_window(np.full(100, val))
            labels = _make_labels_df()
            result = extract_proportions_mean([w], labels)
            assert result["frac_out"][0] in (0.0, 1.0)

    def test_na_limits_produce_nan(self):
        w = _make_window(np.full(100, 70.0))
        labels = _make_labels_df(ULA_Yale_affected_beta=np.nan)
        result = extract_proportions_mean([w], labels)
        assert np.isnan(result["in?"][0])


# ===========================================================================
# Group 4: extract_proportions_smooth
# ===========================================================================


class TestExtractProportionsSmooth:

    @staticmethod
    def _make_smooth_labels(n_minutes, lla=60.0, ula=80.0, mapopt=70.0, r2=0.5):
        """Build a labels df with n_minutes rows, plus start_idx/end_idx."""
        labels = pd.DataFrame(
            {
                "LLA_Yale_affected_beta": [lla] * n_minutes,
                "ULA_Yale_affected_beta": [ula] * n_minutes,
                "MAPopt_Yale_affected_beta": [mapopt] * n_minutes,
                "Yale_R2full_affected": [r2] * n_minutes,
                "start_idx": [0] * n_minutes,
                "end_idx": [n_minutes - 1] * n_minutes,
            }
        )
        return labels

    def test_all_inside(self):
        n_minutes = 5
        samples_per_min = 60
        w = _make_window(np.full(n_minutes * samples_per_min, 70.0))
        labels = self._make_smooth_labels(n_minutes)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        ref = [0]  # index into labels
        result = extract_proportions_smooth([w], labels, 0.0, ref, config)
        assert result["in?"][0] == 1.0
        assert result["frac_out"][0] == pytest.approx(0.0)

    def test_all_outside(self):
        n_minutes = 5
        samples_per_min = 60
        w = _make_window(np.full(n_minutes * samples_per_min, 50.0))
        labels = self._make_smooth_labels(n_minutes)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        ref = [0]
        result = extract_proportions_smooth([w], labels, 0.0, ref, config)
        assert result["in?"][0] == 0.0
        assert result["frac_out"][0] == pytest.approx(1.0)

    def test_threshold_boundary(self):
        # 5 minutes, 2 outside, 3 inside -> frac_out = 0.4 <= 0.46 -> in? = True
        n_minutes = 5
        samples_per_min = 60
        outside = np.full(2 * samples_per_min, 50.0)
        inside = np.full(3 * samples_per_min, 70.0)
        w = _make_window(np.concatenate([outside, inside]))
        labels = self._make_smooth_labels(n_minutes)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        ref = [0]
        result = extract_proportions_smooth([w], labels, 0.0, ref, config)
        assert result["frac_out"][0] == pytest.approx(0.4)
        assert result["in?"][0] == 1.0  # 0.4 <= 0.46

    def test_r2_filter(self):
        n_minutes = 5
        samples_per_min = 60
        w = _make_window(np.full(n_minutes * samples_per_min, 70.0))
        labels = self._make_smooth_labels(n_minutes, r2=0.1)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.5, percentage=0.0)
        ref = [0]
        result = extract_proportions_smooth([w], labels, 0.0, ref, config)
        assert np.isnan(result["in?"][0])

    def test_all_nan_subwindow(self):
        n_minutes = 5
        samples_per_min = 60
        vals = np.full(n_minutes * samples_per_min, 70.0)
        # Make one entire minute NaN
        vals[0:samples_per_min] = np.nan
        w = _make_window(vals)
        labels = self._make_smooth_labels(n_minutes)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        ref = [0]
        result = extract_proportions_smooth([w], labels, 0.0, ref, config)
        assert np.isnan(result["in?"][0])


# ===========================================================================
# Group 5: get_window
# ===========================================================================


class TestGetWindow:
    """Window boundary computation and gap overlap filtering."""

    @staticmethod
    def _setup(window_s=60, window_index=59, percentage=0.5):
        """Build minimal synthetic data for get_window.

        Creates one segment from 0 to 600s at 1Hz (600 samples),
        with labels every 60s starting at 60s.
        """
        freq = 1.0
        n_samples = 600
        data = np.arange(n_samples, dtype=float)

        index = pd.DataFrame(
            {
                "starttime": [np.uint64(0)],
                "endtime": [np.uint64(600_000_000)],
                "frequency": [freq],
                "startidx": [np.uint64(0)],
                "length": [np.uint64(n_samples)],
            }
        )

        # Labels at 60, 120, 180, ..., 540s (in microseconds)
        label_times = np.arange(60, 600, 60, dtype=np.uint64) * 1_000_000
        coords = pd.DataFrame(
            {
                "DateTime": label_times,
                "segment": np.zeros(len(label_times), dtype=int),
            }
        )

        config = SimpleNamespace(percentage=percentage)
        return data, index, coords, window_index, window_s, config

    def test_window_fully_inside_segment(self):
        data, index, coords, wi, ws, config = self._setup()
        df, clean = get_window(data, index, coords, wi, ws, config)
        # All windows should be inside (segment is big enough)
        assert len(df) > 0
        # overlap_len should be 0 for windows fully inside
        for row in df:
            assert row[2] == 0  # overlap_len

    def test_window_exceeds_start_filtered(self):
        # Label at t=30s with window_index=59 (look back 60s) -> window starts at -30s
        # This exceeds segment start -> overlap > 0
        data, index, coords, _, ws, config = self._setup(percentage=0.0)
        # Override coords to have a label at t=30s
        coords = pd.DataFrame(
            {
                "DateTime": [np.uint64(30_000_000)],
                "segment": [0],
            }
        )
        config.percentage = 0.0  # filter everything with any overlap
        df, clean = get_window(data, index, coords, 59, ws, config)
        # With percentage=0, any overlap filters the window
        assert len(df) == 0

    def test_no_filter_with_high_percentage(self):
        # Same as above but allow 100% overlap -> window kept
        data, index, coords, _, ws, config = self._setup(percentage=1.0)
        coords = pd.DataFrame(
            {
                "DateTime": [np.uint64(30_000_000)],
                "segment": [0],
            }
        )
        config.percentage = 1.0
        df, clean = get_window(data, index, coords, 59, ws, config)
        assert len(df) == 1

    def test_start_end_idx_nonnegative(self):
        data, index, coords, wi, ws, config = self._setup()
        df, clean = get_window(data, index, coords, wi, ws, config)
        for row in df:
            assert row[0] >= 0  # w_start_idx
            assert row[1] >= row[0]  # w_end_idx >= w_start_idx

    def test_label_start_end_idx(self):
        data, index, coords, wi, ws, config = self._setup()
        df, clean = get_window(data, index, coords, wi, ws, config)
        # start_idx and end_idx define the range of label rows in the window
        expected_span = ws // 60  # number of 1-minute labels in the window
        for _, row in clean.iterrows():
            assert row["end_idx"] - row["start_idx"] + 1 == expected_span


# ===========================================================================
# Group 6: filter_na and impute
# ===========================================================================


class TestFilterNaAndImpute:

    def test_under_threshold_kept(self):
        # 20% NaN < 25% threshold -> kept
        w = np.ones(100)
        w[:20] = np.nan
        result = filter_na(w)
        assert result is not None

    def test_over_threshold_rejected(self):
        # 30% NaN > 25% threshold -> None
        w = np.ones(100)
        w[:30] = np.nan
        assert filter_na(w) is None

    def test_exact_threshold(self):
        # Exactly 25% NaN -> > is strict, so 25% is NOT rejected
        w = np.ones(100)
        w[:25] = np.nan
        assert filter_na(w) is not None

    def test_no_nans_passthrough(self):
        w = np.array([1.0, 2.0, 3.0])
        result = filter_na(w)
        np.testing.assert_array_equal(result, w)

    def test_impute_linear_interpolation(self):
        w = np.array([1.0, np.nan, 3.0])
        result = impute(w)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_impute_edge_nan(self):
        # np.interp clamps at edges
        w = np.array([np.nan, 2.0, 3.0])
        result = impute(w)
        np.testing.assert_allclose(result, [2.0, 2.0, 3.0])

    def test_impute_no_nans_passthrough(self):
        w = np.array([1.0, 2.0, 3.0])
        result = impute(w)
        np.testing.assert_array_equal(result, w)

    def test_pipeline_produces_no_nans(self):
        """filter_na + impute -> output is None or NaN-free."""
        w = np.ones(100)
        w[10:20] = np.nan  # 10% NaN
        filtered = filter_na(w)
        assert filtered is not None
        imputed = impute(filtered)
        assert not np.isnan(imputed).any()


# ===========================================================================
# Group 7: stride_filter
# ===========================================================================


class TestStrideFilter:

    def test_basic_stride(self):
        # Labels every 60s starting at a large timestamp, window_s = 300
        # Start at 1_000s so the built-in assertion (datetime[0] - 0 > window_us) passes
        base = 1_000_000_000  # 1000s in us
        times = np.arange(0, 3600, 60, dtype=np.uint64) * 1_000_000 + base
        labels = pd.DataFrame({"datetime": times})
        df = np.arange(len(labels))
        result_labels, result_df = stride_filter(labels, df, window_s=300)
        diffs = np.diff(result_labels.datetime.values)
        assert (diffs > 300 * 1e6).all()

    def test_keeps_first(self):
        # Start timestamps large enough to pass the built-in assertion
        base = 1_000_000_000
        times = np.array([0, 100, 200], dtype=np.uint64) * 1_000_000 + base
        labels = pd.DataFrame({"datetime": times})
        df = np.arange(3)
        result_labels, _ = stride_filter(labels, df, window_s=60)
        assert result_labels.datetime.iloc[0] == base

    def test_df_alignment(self):
        base = 1_000_000_000
        times = np.arange(0, 600, 60, dtype=np.uint64) * 1_000_000 + base
        labels = pd.DataFrame({"datetime": times})
        df = np.arange(10) * 10
        result_labels, result_df = stride_filter(labels, df, window_s=300)
        assert len(result_labels) == len(result_df)

    def test_single_label(self):
        # Single label with large enough timestamp
        labels = pd.DataFrame({"datetime": [np.uint64(1_000_000_000)]})
        df = np.array([42])
        result_labels, result_df = stride_filter(labels, df, window_s=60)
        assert len(result_labels) == 1


# ===========================================================================
# Group 8: reshape_flat_windows
# ===========================================================================


class TestReshapeFlatWindows:

    def test_channel_order_preserved(self):
        """Flat array with channel-specific values must maintain order after reshape.

        generate_final concatenates as [ch0_t0..ch0_tT, ch1_t0..ch1_tT, ...]
        reshape does .reshape(N, C, T) then transpose -> (N, T, C)
        So result[:, :, i] should be all (i+1).
        """
        n_samples = 10
        n_channels = 5
        timesteps = 60
        flat = np.zeros((n_samples, n_channels * timesteps))
        for c in range(n_channels):
            flat[:, c * timesteps : (c + 1) * timesteps] = c + 1

        result = reshape_flat_windows(flat, n_channels)
        assert result.shape == (n_samples, timesteps, n_channels)
        for c in range(n_channels):
            np.testing.assert_array_equal(result[:, :, c], c + 1)

    def test_not_divisible_raises(self):
        flat = np.zeros((10, 301))
        with pytest.raises(ValueError, match="not divisible"):
            reshape_flat_windows(flat, 5)

    def test_output_shape(self):
        flat = np.zeros((20, 300))
        result = reshape_flat_windows(flat, 5)
        assert result.shape == (20, 60, 5)

    def test_single_channel(self):
        flat = np.arange(600).reshape(10, 60).astype(float)
        result = reshape_flat_windows(flat, 1)
        assert result.shape == (10, 60, 1)
        np.testing.assert_array_equal(result[:, :, 0], flat)


# ===========================================================================
# Group 9: make_grouped_split
# ===========================================================================


class TestMakeGroupedSplit:

    def test_no_patient_overlap(self):
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
        groups = np.array(["A", "A", "A", "B", "B", "C", "C", "D", "D", "D"])
        train_idx, val_idx = make_grouped_split(y, groups, n_splits=2)
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        assert train_patients & val_patients == set()

    def test_all_indices_covered(self):
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
        groups = np.array(["A", "A", "A", "B", "B", "C", "C", "D", "D", "D"])
        train_idx, val_idx = make_grouped_split(y, groups, n_splits=2)
        all_idx = sorted(list(train_idx) + list(val_idx))
        assert all_idx == list(range(len(y)))

    def test_regression_mode(self):
        y = np.random.randn(20)
        groups = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)
        train_idx, val_idx = make_grouped_split(
            y, groups, n_splits=2, task="regression"
        )
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        assert train_patients & val_patients == set()

    def test_too_few_patients_raises(self):
        y = np.array([0, 1])
        groups = np.array(["A", "A"])
        with pytest.raises(ValueError, match="at least 2"):
            make_grouped_split(y, groups, n_splits=2)


# ===========================================================================
# Group 10: Pipeline invariants
# ===========================================================================


class TestPipelineInvariants:

    def test_labels_binary_or_nan_count(self):
        """extract_proportions_count produces only 0.0, 1.0, or NaN for in?."""
        windows = [
            _make_window(np.full(100, 70.0)),
            _make_window(np.full(100, 50.0)),
            _make_window(np.full(100, np.nan)),
        ]
        labels = pd.DataFrame(
            {
                "LLA_Yale_affected_beta": [60.0, 60.0, 60.0],
                "ULA_Yale_affected_beta": [80.0, 80.0, 80.0],
                "MAPopt_Yale_affected_beta": [70.0, 70.0, 70.0],
            }
        )
        result = extract_proportions_count(windows, labels)
        for val in result["in?"]:
            assert val in (0.0, 1.0) or np.isnan(val)

    def test_labels_binary_or_nan_mean(self):
        windows = [
            _make_window(np.full(100, 70.0)),
            _make_window(np.full(100, 50.0)),
        ]
        labels = pd.DataFrame(
            {
                "LLA_Yale_affected_beta": [60.0, 60.0],
                "ULA_Yale_affected_beta": [80.0, 80.0],
                "MAPopt_Yale_affected_beta": [70.0, 70.0],
            }
        )
        result = extract_proportions_mean(windows, labels)
        for val in result["in?"]:
            assert val in (0.0, 1.0) or np.isnan(val)

    def test_proportion_out_is_pessimistic_upper_bound(self):
        """proportion_out = 1 - proportion_in is a pessimistic upper bound.

        NaN samples are dropped before computing inside status, so they
        contribute to proportion_out. This is intentional: the margin check
        on line 325 uses proportion_na to account for the uncertainty.

        In a window with 80 inside, 10 outside, 10 NaN (total_length=100):
        - proportion_in = 80/100 = 0.80
        - proportion_out = 0.20 (includes 10 truly outside + 10 NaN)
        - proportion_na = 0.10
        - margin = 0.80 - 0.20 = 0.60 >= 0.10 -> label assigned
        """
        inside = np.full(80, 70.0)
        outside = np.full(10, 50.0)
        nans = np.full(10, np.nan)
        vals = np.concatenate([inside, outside, nans])
        w = _make_window(vals, overlap_len=0, total_length=100)
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)

        assert result["frac_out"][0] == pytest.approx(0.20)
        # Label is assigned because margin (0.60) exceeds uncertainty (0.10)
        assert result["in?"][0] == 1.0

    def test_robust_floor_ceil(self):
        # Floating point edge cases
        assert robust_floor(2.9999999) == 3.0
        assert robust_ceil(3.0000001) == 3.0
        # Normal cases
        assert robust_floor(2.5) == 2.0
        assert robust_ceil(2.5) == 3.0


# ===========================================================================
# Group 11: _bad_frac (physiological value filtering during downsampling)
# ===========================================================================


class TestBadFrac:
    """Tests for _bad_frac() in data_extract.py — NaN and physiological range check."""

    @staticmethod
    def _make_3d(values):
        """Wrap a list of group values into a (1, 1, G) dask array."""
        arr = np.array([[values]], dtype=float)
        return da.from_array(arr)

    def test_all_valid_non_abp(self):
        reshaped = self._make_3d([70.0, 80.0, 90.0, 60.0])
        result = _bad_frac(reshaped, is_abp=False).compute()
        assert result[0, 0] == pytest.approx(0.0)

    def test_all_nan_non_abp(self):
        reshaped = self._make_3d([np.nan, np.nan, np.nan, np.nan])
        result = _bad_frac(reshaped, is_abp=False).compute()
        assert result[0, 0] == pytest.approx(1.0)

    def test_partial_nan_non_abp(self):
        reshaped = self._make_3d([np.nan, 70.0, 80.0, 90.0])
        result = _bad_frac(reshaped, is_abp=False).compute()
        assert result[0, 0] == pytest.approx(0.25)

    def test_abp_out_of_range_counted(self):
        # 10 < 20 (below) and 250 > 200 (above) -> 2/4 bad
        reshaped = self._make_3d([10.0, 70.0, 250.0, 80.0])
        result = _bad_frac(reshaped, is_abp=True).compute()
        assert result[0, 0] == pytest.approx(0.5)

    def test_abp_nan_and_out_of_range(self):
        # NaN + 10.0 (below range) -> 2/4 bad
        reshaped = self._make_3d([np.nan, 10.0, 70.0, 80.0])
        result = _bad_frac(reshaped, is_abp=True).compute()
        assert result[0, 0] == pytest.approx(0.5)

    def test_abp_all_physiological(self):
        reshaped = self._make_3d([80.0, 80.0, 80.0, 80.0])
        result = _bad_frac(reshaped, is_abp=True).compute()
        assert result[0, 0] == pytest.approx(0.0)

    def test_non_abp_ignores_range(self):
        # Out-of-ABP-range values are NOT flagged for non-ABP variables
        reshaped = self._make_3d([10.0, 250.0, 70.0, 80.0])
        result = _bad_frac(reshaped, is_abp=False).compute()
        assert result[0, 0] == pytest.approx(0.0)

    def test_threshold_boundary(self):
        # Exactly 25% bad -> 0.25 is NOT > 0.25 (strict >)
        reshaped = self._make_3d([np.nan, 70.0, 80.0, 90.0])
        frac = _bad_frac(reshaped, is_abp=False).compute()[0, 0]
        assert frac == pytest.approx(0.25)
        assert not (frac > ABP_MAX_BAD_FRAC)

    def test_abp_boundary_values_are_valid(self):
        # Exactly 20.0 and 200.0 should be considered physiological (>= and <=)
        reshaped = self._make_3d([20.0, 200.0, 100.0, 100.0])
        result = _bad_frac(reshaped, is_abp=True).compute()
        assert result[0, 0] == pytest.approx(0.0)

    def test_multiple_groups(self):
        # (1, 3, 4) -> 3 groups of 4 values each
        arr = np.array([[[70.0, 80.0, 90.0, 60.0],
                         [np.nan, np.nan, np.nan, np.nan],
                         [np.nan, 70.0, 80.0, 90.0]]], dtype=float)
        reshaped = da.from_array(arr)
        result = _bad_frac(reshaped, is_abp=False).compute()
        np.testing.assert_allclose(result[0], [0.0, 1.0, 0.25])


# ===========================================================================
# Group 12: ar_class from extract_proportions_count
# ===========================================================================


class TestArClassCount:
    """ar_class labels from extract_proportions_count."""

    def test_ar_class_in(self):
        w = _make_window(np.full(100, 70.0))
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        assert result["ar_class"][0] == 1

    def test_ar_class_below(self):
        w = _make_window(np.full(100, 50.0))
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        assert result["ar_class"][0] == 0

    def test_ar_class_above(self):
        w = _make_window(np.full(100, 90.0))
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        assert result["ar_class"][0] == 2

    def test_ar_class_nan_limits(self):
        w = _make_window(np.full(100, 70.0))
        labels = _make_labels_df(LLA_Yale_affected_beta=np.nan)
        result = extract_proportions_count([w], labels)
        assert np.isnan(result["ar_class"][0])

    def test_ar_class_ambiguous_nan(self):
        # Same ambiguous case: 52 inside, 48 outside, 10% gap overlap
        vals = np.concatenate([np.full(52, 70.0), np.full(48, 50.0)])
        w = _make_window(vals, overlap_len=10, total_length=100)
        labels = _make_labels_df()
        result = extract_proportions_count([w], labels)
        assert np.isnan(result["ar_class"][0])

    def test_ar_class_consistent_with_in(self):
        """ar_class == 1 iff in? == 1.0 for all non-NaN results."""
        windows = [
            _make_window(np.full(100, 70.0)),  # inside
            _make_window(np.full(100, 50.0)),  # below
            _make_window(np.full(100, 90.0)),  # above
        ]
        labels = pd.DataFrame({
            "LLA_Yale_affected_beta": [60.0] * 3,
            "ULA_Yale_affected_beta": [80.0] * 3,
            "MAPopt_Yale_affected_beta": [70.0] * 3,
        })
        result = extract_proportions_count(windows, labels)
        for i in range(3):
            if not np.isnan(result["in?"][i]) and not np.isnan(result["ar_class"][i]):
                assert (result["ar_class"][i] == 1) == (result["in?"][i] == 1.0)


# ===========================================================================
# Group 13: ar_class from extract_proportions_mean
# ===========================================================================


class TestArClassMean:
    """ar_class labels from extract_proportions_mean."""

    def test_ar_class_in(self):
        w = _make_window(np.full(100, 70.0))
        labels = _make_labels_df()
        result = extract_proportions_mean([w], labels)
        assert result["ar_class"][0] == 1

    def test_ar_class_below(self):
        w = _make_window(np.full(100, 50.0))
        labels = _make_labels_df()
        result = extract_proportions_mean([w], labels)
        assert result["ar_class"][0] == 0

    def test_ar_class_above(self):
        w = _make_window(np.full(100, 90.0))
        labels = _make_labels_df()
        result = extract_proportions_mean([w], labels)
        assert result["ar_class"][0] == 2

    def test_ar_class_nan_window(self):
        w = _make_window(np.full(100, np.nan))
        labels = _make_labels_df()
        result = extract_proportions_mean([w], labels)
        assert np.isnan(result["ar_class"][0])

    def test_ar_class_consistent_with_in(self):
        windows = [
            _make_window(np.full(100, 70.0)),
            _make_window(np.full(100, 50.0)),
            _make_window(np.full(100, 90.0)),
        ]
        labels = pd.DataFrame({
            "LLA_Yale_affected_beta": [60.0] * 3,
            "ULA_Yale_affected_beta": [80.0] * 3,
            "MAPopt_Yale_affected_beta": [70.0] * 3,
        })
        result = extract_proportions_mean(windows, labels)
        for i in range(3):
            if not np.isnan(result["in?"][i]) and not np.isnan(result["ar_class"][i]):
                assert (result["ar_class"][i] == 1) == (result["in?"][i] == 1.0)


# ===========================================================================
# Group 14: ar_class from extract_proportions_smooth
# ===========================================================================


class TestArClassSmooth:
    """ar_class labels from extract_proportions_smooth."""

    @staticmethod
    def _make_smooth_labels(n_minutes, lla=60.0, ula=80.0, mapopt=70.0, r2=0.5):
        labels = pd.DataFrame({
            "LLA_Yale_affected_beta": [lla] * n_minutes,
            "ULA_Yale_affected_beta": [ula] * n_minutes,
            "MAPopt_Yale_affected_beta": [mapopt] * n_minutes,
            "Yale_R2full_affected": [r2] * n_minutes,
            "start_idx": [0] * n_minutes,
            "end_idx": [n_minutes - 1] * n_minutes,
        })
        return labels

    def test_ar_class_in(self):
        n_minutes = 5
        w = _make_window(np.full(n_minutes * 60, 70.0))
        labels = self._make_smooth_labels(n_minutes)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        result = extract_proportions_smooth([w], labels, 0.0, [0], config)
        assert result["ar_class"][0] == 1

    def test_ar_class_below(self):
        n_minutes = 5
        w = _make_window(np.full(n_minutes * 60, 50.0))
        labels = self._make_smooth_labels(n_minutes)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        result = extract_proportions_smooth([w], labels, 0.0, [0], config)
        assert result["ar_class"][0] == 0

    def test_ar_class_above(self):
        n_minutes = 5
        w = _make_window(np.full(n_minutes * 60, 90.0))
        labels = self._make_smooth_labels(n_minutes)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        result = extract_proportions_smooth([w], labels, 0.0, [0], config)
        assert result["ar_class"][0] == 2

    def test_ar_class_r2_filter(self):
        n_minutes = 5
        w = _make_window(np.full(n_minutes * 60, 70.0))
        labels = self._make_smooth_labels(n_minutes, r2=0.1)
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.5, percentage=0.0)
        result = extract_proportions_smooth([w], labels, 0.0, [0], config)
        assert np.isnan(result["ar_class"][0])

    def test_ar_class_consistent_with_in(self):
        n_minutes = 5
        samples = n_minutes * 60
        windows = [
            _make_window(np.full(samples, 70.0)),
            _make_window(np.full(samples, 50.0)),
            _make_window(np.full(samples, 90.0)),
        ]
        labels = pd.DataFrame({
            "LLA_Yale_affected_beta": [60.0] * n_minutes,
            "ULA_Yale_affected_beta": [80.0] * n_minutes,
            "MAPopt_Yale_affected_beta": [70.0] * n_minutes,
            "Yale_R2full_affected": [0.5] * n_minutes,
            "start_idx": [0] * n_minutes,
            "end_idx": [n_minutes - 1] * n_minutes,
        })
        config = SimpleNamespace(smooth_frac=0.46, r2_threshold=0.0, percentage=0.0)
        for w in windows:
            result = extract_proportions_smooth([w], labels, 0.0, [0], config)
            in_val = result["in?"][0]
            ar_val = result["ar_class"][0]
            if not np.isnan(in_val) and not np.isnan(ar_val):
                assert (ar_val == 1) == (in_val == 1.0)
