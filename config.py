"""
config.py - one place for every setting the project shares.

Every other file imports from here, so changing a value (like the seed)
changes it everywhere at once.
"""

from pathlib import Path

# ---- Reproducibility -------------------------------------------------------
# Same seed -> same random bits, same noise, same results on every run.
SEED = 42

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"        # generated data (kept out of Git)
REPORTS_DIR = ROOT / "reports"  # generated HTML reports (kept out of Git)
THRESHOLDS_FILE = DATA_DIR / "thresholds.json"

# ---- Simulator convention --------------------------------------------------
# All noise levels in this project are given as Eb/N0 in dB
# (energy per BIT divided by noise power density).
# SNR per symbol (Es/N0) = Eb/N0 + 10*log10(bits_per_symbol).
BITS_PER_SYMBOL = {"QPSK": 2, "16QAM": 4}

# ---- Test lab (Phase 3) ----------------------------------------------------
WATCHDOG_S = 60.0          # a test running longer than this (lab-seconds) times out
FLAKY_RATE = 0.02          # chance any run fails for a harness reason (no real bug)
HINT_PROBABILITY = 0.7     # chance an active fault leaves its hint line in a log
NOISE_WARNING_RATE = 0.25  # chance a log gets one harmless, misleading warning
THRESHOLD_MARGIN = 1.5     # spec limit = margin x worst BER of the golden build
CALIBRATION_RUNS = 5       # golden-build runs per test used to set thresholds