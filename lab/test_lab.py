"""
tests/test_lab.py - checks the test lab: catalog, golden build, fault
signatures, flakiness, logs and ground truth.

These tests protect the most important property of the whole project:
each injected fault breaks the tests it should - and only those - and the
ground truth is kept apart from what the model and agent are allowed to see.
"""

import pytest

from lab.runner import run_test
from lab.testcases import load_tests
from sim.device import Device
from sim.faults import FAULTS, FLAKY_LINE


@pytest.fixture(scope="module")
def tests():
    return load_tests()


def run_all(tests, device, flaky_rate=0.0):
    """Run every test on one build; return a list of (testcase, result, truth)."""
    out = []
    for i, tc in enumerate(tests):
        result, truth = run_test(tc, device, i, flaky_rate=flaky_rate)
        out.append((tc, result, truth))
    return out


def failed(rows, pick=lambda tc: True):
    """How many tests matching `pick` did not pass, and how many matched."""
    mine = [r for tc, r, _ in rows if pick(tc)]
    return sum(r["status"] != "pass" for r in mine), len(mine)


# ---- catalog ---------------------------------------------------------------
def test_catalog_has_52_unique_tests(tests):
    ids = [t.test_id for t in tests]
    assert len(ids) == 52 and len(set(ids)) == 52
    assert sum(t.kind == "ber" for t in tests) == 48
    assert sum(t.kind == "stress" for t in tests) == 4


def test_every_test_has_a_positive_threshold(tests):
    assert all(t.threshold is not None and t.threshold > 0 for t in tests)


def test_components_follow_test_design(tests):
    for t in tests:
        assert ("equalizer" in t.components) == (t.channel == "rayleigh")
        assert ("scheduler" in t.components) == (t.length in ("long", "stress"))


# ---- golden build ------------------------------------------------------------
def test_golden_build_passes_everything(tests):
    rows = run_all(tests, Device(1))
    assert failed(rows) == (0, 52)


# ---- fault signatures: each bug breaks what it should, and only that ---------
def test_mapper_fault_breaks_only_16qam(tests):
    rows = run_all(tests, Device(1, {"mapper_scale"}))
    assert failed(rows, lambda t: t.modulation == "QPSK")[0] == 0
    bad, n = failed(rows, lambda t: t.modulation == "16QAM")
    assert bad >= 0.9 * n


def test_equalizer_fault_breaks_only_fading(tests):
    rows = run_all(tests, Device(1, {"eq_disabled"}))
    assert failed(rows, lambda t: t.channel == "awgn")[0] == 0
    bad, n = failed(rows, lambda t: t.channel == "rayleigh")
    assert bad == n


def test_scheduler_fault_only_times_out_long_runs(tests):
    rows = run_all(tests, Device(1, {"sched_slow"}))
    for tc, r, _ in rows:
        if tc.length in ("long", "stress"):
            assert r["status"] == "timeout"
        else:
            assert r["status"] == "pass"


def test_sync_fault_spares_qpsk_fading(tests):
    rows = run_all(tests, Device(1, {"sync_phase"}))
    assert failed(rows, lambda t: t.modulation == "QPSK" and t.channel == "rayleigh")[0] == 0
    bad, n = failed(rows, lambda t: t.modulation == "16QAM" and t.channel == "awgn")
    assert bad == n


def test_rf_fault_breaks_all_awgn_tests(tests):
    rows = run_all(tests, Device(1, {"rf_gain_mismatch"}))
    bad, n = failed(rows, lambda t: t.channel == "awgn")
    assert bad == n


# ---- ground truth --------------------------------------------------------------
def test_truth_names_the_right_fault_when_two_are_active(tests):
    rows = run_all(tests, Device(5, {"eq_disabled", "mapper_scale"}))
    for tc, r, truth in rows:
        if r["status"] == "pass":
            assert truth["true_causes"] == []
        elif tc.modulation == "QPSK":            # mapper bug cannot reach QPSK
            assert truth["true_causes"] == ["eq_disabled"]
        elif tc.channel == "awgn":               # equaliser is unused on AWGN
            assert truth["true_causes"] == ["mapper_scale"]


def test_result_never_contains_the_answer(tests):
    result, truth = run_test(tests[0], Device(3, {"rf_gain_mismatch"}), 0, 0.0)
    assert "true_causes" not in result
    assert "true_causes" in truth


# ---- flakiness ---------------------------------------------------------------
def test_flaky_failures_are_labelled_flaky(tests):
    rows = run_all(tests[:10], Device(1), flaky_rate=1.0)
    for tc, r, truth in rows:
        assert r["status"] == "fail"
        assert truth["true_causes"] == ["flaky"]
        assert FLAKY_LINE in r["log"]


# ---- logs ----------------------------------------------------------------------
def test_mapper_hint_never_appears_in_qpsk_logs(tests):
    rows = run_all(tests, Device(1, {"mapper_scale"}))
    hint = FAULTS["mapper_scale"]["hint"]
    for tc, r, _ in rows:
        if tc.modulation == "QPSK":
            assert hint not in r["log"]


def test_failing_log_states_why(tests):
    rows = run_all(tests, Device(1, {"eq_disabled", "sched_slow"}))
    for tc, r, _ in rows:
        if r["status"] == "fail":
            assert "FAIL: ber above spec" in r["log"]
        if r["status"] == "timeout":
            assert "TIMEOUT: watchdog expired" in r["log"]
            assert r["ber"] is None


def test_runs_are_reproducible(tests):
    d = Device(9, {"sync_phase"})
    assert run_test(tests[20], d, 20) == run_test(tests[20], d, 20)


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError, match="unknown fault"):
        Device(1, {"gremlins"})