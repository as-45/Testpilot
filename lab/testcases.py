"""
lab/testcases.py - the 52 automated tests of the simulated validation lab.

Each test is a spec check: "send N bits with this modulation, channel and
Eb/N0; the measured BER must stay at or below the limit, and the run must
finish before the watchdog".

The BER limits are not guessed.  We run the healthy ("golden") build several
times, take the worst BER each test ever showed, and allow THRESHOLD_MARGIN
times that.  This is how real validation sets limits from a known-good
reference.  Run this file to (re)compute them:

    python -m lab.testcases
"""

import json
from dataclasses import dataclass, replace

import numpy as np

from config import (
    CALIBRATION_RUNS,
    SEED,
    THRESHOLD_MARGIN,
    THRESHOLDS_FILE,
)
from sim.channel import simulate_link

# Eb/N0 points per (modulation, channel): chosen so the healthy build makes
# a measurable number of errors at every point (no zero-error tests).
EBN0_POINTS = {
    ("QPSK", "awgn"): (2, 3, 4, 5, 6, 7),
    ("16QAM", "awgn"): (6, 7, 8, 9, 10, 11),
    ("QPSK", "rayleigh"): (6, 8, 10, 12, 14, 16),
    ("16QAM", "rayleigh"): (10, 12, 14, 16, 18, 20),
}
LENGTHS = {"short": 20_000, "long": 100_000}     # bits per run
BASE_COST_S = {"short": 8.0, "long": 32.0}       # lab-seconds per run
STRESS_BITS = 200_000
STRESS_COST_S = 40.0
RAYLEIGH_COST_FACTOR = 1.2                       # fading runs are a bit slower


@dataclass(frozen=True)
class TestCase:
    __test__ = False          # tell pytest this is not a test class

    test_id: str
    kind: str                 # "ber" or "stress"
    modulation: str           # "QPSK" or "16QAM"
    channel: str              # "awgn" or "rayleigh"
    ebn0_db: float
    length: str               # "short", "long" or "stress"
    n_bits: int
    cost_s: float             # nominal lab-seconds (healthy build)
    components: tuple         # components this test exercises (from its design)
    threshold: float | None = None   # BER limit, filled in by calibration


def _components_for(channel: str, length: str) -> tuple:
    """Which parts of the modem a test exercises, decided by its design."""
    comps = ["rf_frontend", "mapper", "sync"]          # every BER test
    if channel == "rayleigh":
        comps.append("equalizer")                       # only fading needs it
    if length in ("long", "stress"):
        comps.append("scheduler")                       # long runs stress timing
    return tuple(comps)


def build_test_catalog() -> list[TestCase]:
    """The 48 BER tests + 4 stress tests, in a fixed order (no thresholds yet)."""
    tests = []
    for (mod, ch), points in EBN0_POINTS.items():
        for ebn0 in points:
            for length, n_bits in LENGTHS.items():
                cost = BASE_COST_S[length]
                if ch == "rayleigh":
                    cost *= RAYLEIGH_COST_FACTOR
                tests.append(TestCase(
                    test_id=f"ber_{mod.lower()}_{ch}_eb{ebn0:02d}_{length}",
                    kind="ber", modulation=mod, channel=ch, ebn0_db=float(ebn0),
                    length=length, n_bits=n_bits, cost_s=round(cost, 1),
                    components=_components_for(ch, length),
                ))
    # stress tests: the highest Eb/N0 point of each pair, many more bits
    for (mod, ch), points in EBN0_POINTS.items():
        cost = STRESS_COST_S * (RAYLEIGH_COST_FACTOR if ch == "rayleigh" else 1.0)
        tests.append(TestCase(
            test_id=f"stress_{mod.lower()}_{ch}",
            kind="stress", modulation=mod, channel=ch, ebn0_db=float(points[-1]),
            length="stress", n_bits=STRESS_BITS, cost_s=round(cost, 1),
            components=_components_for(ch, "stress"),
        ))
    return tests


def calibrate_thresholds(tests: list[TestCase]) -> dict[str, float]:
    """Run the golden build CALIBRATION_RUNS times per test; limit = margin x worst."""
    thresholds = {}
    for idx, tc in enumerate(tests):
        worst = 0.0
        for run in range(CALIBRATION_RUNS):
            # calibration seeds live in their own space, never reused by builds
            seed = int(np.random.SeedSequence([SEED, 999_999, run, idx])
                       .generate_state(1)[0])
            r = simulate_link(tc.modulation, tc.channel, tc.ebn0_db, tc.n_bits, seed)
            worst = max(worst, r["ber"])
        if worst == 0.0:
            raise RuntimeError(
                f"{tc.test_id}: golden build made zero errors; pick a lower Eb/N0"
            )
        thresholds[tc.test_id] = THRESHOLD_MARGIN * worst
    return thresholds


def save_thresholds(thresholds: dict[str, float]) -> None:
    THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_FILE.write_text(json.dumps(thresholds, indent=2))


def load_tests() -> list[TestCase]:
    """The full catalog with thresholds; calibrates once and caches to data/."""
    tests = build_test_catalog()
    if THRESHOLDS_FILE.exists():
        thresholds = json.loads(THRESHOLDS_FILE.read_text())
        if set(thresholds) != {t.test_id for t in tests}:
            thresholds = None          # catalog changed -> recalibrate
    else:
        thresholds = None
    if thresholds is None:
        thresholds = calibrate_thresholds(tests)
        save_thresholds(thresholds)
    return [replace(t, threshold=thresholds[t.test_id]) for t in tests]


if __name__ == "__main__":
    tests = build_test_catalog()
    print(f"Calibrating {len(tests)} tests on the golden build "
          f"({CALIBRATION_RUNS} runs each)...")
    th = calibrate_thresholds(tests)
    save_thresholds(th)
    print(f"Saved thresholds to {THRESHOLDS_FILE}\n")
    print(f"{'test_id':<34} {'cost_s':>6}  {'threshold':>9}  components")
    for t in tests:
        print(f"{t.test_id:<34} {t.cost_s:>6}  {th[t.test_id]:>9.2e}  "
              f"{','.join(t.components)}")