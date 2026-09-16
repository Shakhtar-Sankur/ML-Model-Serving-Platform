"""Drift detection, and the cancellation it used to be blind to.

The original flattened every feature into one distribution before running a
single KS test. `test_opposite_shifts_do_not_cancel` is the case that makes that
wrong: feature 0 moves up by the same amount feature 1 moves down, the pooled
distribution barely changes, and a real shift goes unreported.

The data is seeded, so a failure here is a change in behaviour and not a bad
draw.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("scipy")

from ml_serving_platform import DataDriftDetector


N_FEATURES = 4
N_ROWS = 400


def sample(seed, shifts=None):
    """N_ROWS rows of N_FEATURES standard-normal columns, optionally shifted."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(N_ROWS, N_FEATURES))
    for index, amount in (shifts or {}).items():
        data[:, index] += amount
    return data


def test_the_same_distribution_is_not_drift():
    report = DataDriftDetector(sample(1)).detect_drift(sample(2))
    assert not report["drift_detected"], \
        f"drift reported on two draws from one distribution: {report['drifted_features']}"


def test_a_shifted_feature_is_detected_and_named():
    report = DataDriftDetector(sample(1)).detect_drift(sample(2, shifts={2: 1.5}))
    assert report["drift_detected"]
    assert report["drifted_features"] == [2], \
        f"expected feature 2 only, got {report['drifted_features']}"


def test_opposite_shifts_do_not_cancel():
    """The defect: pooled together, +2 on one feature hid -2 on another."""
    report = DataDriftDetector(sample(1)).detect_drift(sample(2, shifts={0: 2.0, 1: -2.0}))
    assert report["drift_detected"], "two shifted features cancelled each other out"
    assert set(report["drifted_features"]) == {0, 1}


def test_the_threshold_is_corrected_for_the_number_of_tests():
    """One test per feature inflates the false-positive rate; Bonferroni divides."""
    report = DataDriftDetector(sample(1)).detect_drift(sample(2))
    assert report["threshold"] == pytest.approx(0.05 / N_FEATURES)


def test_every_feature_is_reported_on():
    report = DataDriftDetector(sample(1)).detect_drift(sample(2, shifts={0: 3.0}))
    assert len(report["per_feature"]) == N_FEATURES
    for entry in report["per_feature"]:
        assert 0.0 <= entry["p_value"] <= 1.0
        assert 0.0 <= entry["ks_statistic"] <= 1.0
    assert report["per_feature"][0]["ks_statistic"] > report["per_feature"][1]["ks_statistic"]


def test_the_means_are_reported_per_feature_not_pooled():
    report = DataDriftDetector(sample(1)).detect_drift(sample(2, shifts={3: 5.0}))
    assert len(report["reference_mean"]) == N_FEATURES
    assert len(report["new_mean"]) == N_FEATURES
    assert report["new_mean"][3] - report["reference_mean"][3] == pytest.approx(5.0, abs=0.3)


def test_a_large_shift_gives_a_smaller_p_value_than_a_small_one():
    small = DataDriftDetector(sample(1)).detect_drift(sample(2, shifts={0: 0.3}))
    large = DataDriftDetector(sample(1)).detect_drift(sample(2, shifts={0: 3.0}))
    assert large["per_feature"][0]["p_value"] < small["per_feature"][0]["p_value"]


def test_image_shaped_data_is_flattened_per_pixel():
    """Inputs arrive as (n, h, w, c); each position is its own feature."""
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(60, 2, 2, 1))
    current = rng.normal(size=(60, 2, 2, 1))
    report = DataDriftDetector(reference).detect_drift(current)
    assert len(report["per_feature"]) == 4
    assert report["threshold"] == pytest.approx(0.05 / 4)


def test_the_report_is_json_serialisable():
    """It is returned from a Flask endpoint; numpy scalars would not serialise."""
    import json
    report = DataDriftDetector(sample(1)).detect_drift(sample(2, shifts={1: 2.0}))
    json.dumps(report)
    assert isinstance(report["drift_detected"], bool)
    assert all(isinstance(f["drifted"], bool) for f in report["per_feature"])
