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

# ---- Simulator convention --------------------------------------------------
# All noise levels in this project are given as Eb/N0 in dB
# (energy per BIT divided by noise power density).
# SNR per symbol (Es/N0) = Eb/N0 + 10*log10(bits_per_symbol).
# For QPSK (2 bits/symbol): Es/N0 = Eb/N0 + 3.01 dB.
BITS_PER_SYMBOL = {"QPSK": 2}