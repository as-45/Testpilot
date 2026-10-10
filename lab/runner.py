"""
lab/runner.py - runs one test on one build and writes its log.

run_test() returns TWO things, kept strictly apart:
  * result - what a real lab would record (status, BER, duration, log).
             This is the only thing the ML model and triage agent may see.
  * truth  - why it really failed (which injected fault, or flaky).
             Used ONLY to score the model and the agent later.

Run this file to see each fault's "signature" - which tests it breaks:
    python -m lab.runner
"""

import numpy as np

from config import FLAKY_RATE, HINT_PROBABILITY, NOISE_WARNING_RATE, SEED, WATCHDOG_S
from lab.testcases import TestCase
from sim.channel import simulate_link
from sim.device import Device
from sim.faults import FAULTS, FLAKY_LINE, NOISE_WARNINGS


def run_seed(build_id: int, test_index: int) -> int:
    """A unique, reproducible seed for every (build, test) pair."""
    return int(np.random.SeedSequence([SEED, build_id, test_index])
               .generate_state(1)[0])


def _execute(tc: TestCase, device: Device, seed: int) -> tuple[str, float | None, float]:
    """
    The measurement itself, with no logging and no flakiness.
    Returns (status, ber, duration_s).
    """
    if tc.threshold is None:
        raise ValueError(f"{tc.test_id} has no threshold; load tests with load_tests()")
    extras = np.random.default_rng([seed, 1])          # timing jitter
    jitter = extras.uniform(0.9, 1.1)
    duration = tc.cost_s * jitter * device.slowdown(tc.components, tc.modulation)
    if duration > WATCHDOG_S:
        return "timeout", None, round(WATCHDOG_S, 1)   # aborted at the watchdog
    r = simulate_link(tc.modulation, tc.channel, tc.ebn0_db, tc.n_bits, seed,
                      device.impairments(tc.components, tc.modulation))
    status = "pass" if r["ber"] <= tc.threshold else "fail"
    return status, r["ber"], round(duration, 1)


def run_test(
    tc: TestCase,
    device: Device,
    test_index: int,
    flaky_rate: float = FLAKY_RATE,
) -> tuple[dict, dict]:
    """Run one test on one build. Returns (result, truth)."""
    seed = run_seed(device.build_id, test_index)
    status, ber, duration = _execute(tc, device, seed)

    logrng = np.random.default_rng([seed, 2])          # log extras + flakiness
    flaky = status == "pass" and logrng.random() < flaky_rate
    if flaky:
        status = "fail"

    # ---- the log: what an engineer would read ---------------------------
    lines = [
        f"[INFO ]  test={tc.test_id} build={device.build_id}",
        f"[INFO ]  config modulation={tc.modulation} channel={tc.channel} "
        f"ebn0_db={tc.ebn0_db:.1f} bits={tc.n_bits}",
    ]
    warnings = []
    for name in device.faults_touching(tc.components, tc.modulation):
        if logrng.random() < HINT_PROBABILITY:
            warnings.append(FAULTS[name]["hint"])
    if logrng.random() < NOISE_WARNING_RATE:
        warnings.append(NOISE_WARNINGS[logrng.integers(len(NOISE_WARNINGS))])
    logrng.shuffle(warnings)
    lines += warnings

    if status == "timeout":
        lines += [
            "[INFO ]  measured ber=n/a (run aborted)",
            f"[ERROR]  TIMEOUT: watchdog expired after {WATCHDOG_S:.1f}s",
        ]
    else:
        lines += [
            f"[INFO ]  measured ber={ber:.3e} threshold={tc.threshold:.3e}",
            f"[INFO ]  duration_s={duration:.1f}",
        ]
        if flaky:
            lines.append(FLAKY_LINE)
        elif status == "fail":
            lines.append(f"[ERROR]  FAIL: ber above spec by {ber / tc.threshold:.1f}x")
        else:
            lines.append("[PASS ]  ber within spec")

    result = {
        "build_id": device.build_id,
        "test_id": tc.test_id,
        "status": status,
        "ber": ber,
        "threshold": tc.threshold,
        "duration_s": duration,
        "log": "\n".join(lines),
    }
    truth = {
        "build_id": device.build_id,
        "test_id": tc.test_id,
        "status": status,
        "true_causes": attribute_failure(tc, device, seed, status, flaky),
    }
    return result, truth


def attribute_failure(tc, device, seed, status, flaky) -> list[str]:
    """
    Ground truth: WHICH fault(s) made this run fail?

    We rerun the same test, same seed, with each active fault on its own.
    A fault that still fails alone is a true cause.  If no single fault does
    it, the failure needed the combination, so all touching faults count.
    """
    if status == "pass":
        return []
    if flaky:
        return ["flaky"]
    candidates = device.faults_touching(tc.components, tc.modulation)
    causes = [
        f for f in candidates
        if _execute(tc, Device(device.build_id, {f}), seed)[0] != "pass"
    ]
    return causes or candidates


if __name__ == "__main__":
    from lab.testcases import load_tests

    tests = load_tests()
    builds = [("golden (no faults)", Device(1, set()))] + [
        (name, Device(1, {name})) for name in FAULTS
    ]
    print(f"Fault signatures: how many of {len(tests)} tests fail on a build "
          f"with ONE fault (flakiness off)\n")
    groups = {
        "QPSK awgn": lambda t: t.modulation == "QPSK" and t.channel == "awgn",
        "16QAM awgn": lambda t: t.modulation == "16QAM" and t.channel == "awgn",
        "QPSK fading": lambda t: t.modulation == "QPSK" and t.channel == "rayleigh",
        "16QAM fading": lambda t: t.modulation == "16QAM" and t.channel == "rayleigh",
    }
    header = f"{'build':<20}" + "".join(f"{g:>14}" for g in groups) + f"{'timeouts':>10}{'total':>7}"
    print(header)
    print("-" * len(header))
    for label, device in builds:
        rows = [(t, run_test(t, device, i, flaky_rate=0.0)[0]) for i, t in enumerate(tests)]
        cells = []
        for pick in groups.values():
            mine = [r for t, r in rows if pick(t)]
            bad = sum(r["status"] != "pass" for r in mine)
            cells.append(f"{bad:>3}/{len(mine):<3}")
        timeouts = sum(r["status"] == "timeout" for _, r in rows)
        total = sum(r["status"] != "pass" for _, r in rows)
        print(f"{label:<20}" + "".join(f"{c:>14}" for c in cells) + f"{timeouts:>10}{total:>7}")

    print("\nExample log (16QAM fading test on a build with eq_disabled):\n")
    tc = next(t for t in tests if t.test_id == "ber_16qam_rayleigh_eb18_long")
    print(run_test(tc, Device(7, {"eq_disabled"}), tests.index(tc), 0.0)[0]["log"])