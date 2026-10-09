"""
sim/channel.py - the radio physics of TestPilot (Phase 1: QPSK over AWGN).

The chain this file simulates:

    random bits -> QPSK symbols -> + noise (AWGN) -> receiver decides -> bits
                                                                         |
                               compare with the bits we sent  <----------+
                               -> BER (bit error rate)

Conventions (keep these fixed for the whole project):
  * Symbols are scaled so their average energy Es = 1.
  * Noise level is given as Eb/N0 in dB (energy per bit / noise density).
  * QPSK carries 2 bits per symbol, so Eb = Es / 2.
"""

import numpy as np
from scipy.special import erfc

QPSK_BITS_PER_SYMBOL = 2


# ---------------------------------------------------------------------------
# 1. Bits
# ---------------------------------------------------------------------------
def generate_bits(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Return n_bits random 0/1 values as a uint8 array."""
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 2. QPSK modulation / demodulation
# ---------------------------------------------------------------------------
def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Turn bits into QPSK symbols (Gray mapping).

    Bits are taken two at a time: the first bit sets the real part (I),
    the second sets the imaginary part (Q).  0 -> +1, 1 -> -1.
    Dividing by sqrt(2) makes every symbol's energy exactly 1.

        bits 00 -> (+1 + 1j)/sqrt2      bits 01 -> (+1 - 1j)/sqrt2
        bits 10 -> (-1 + 1j)/sqrt2      bits 11 -> (-1 - 1j)/sqrt2
    """
    bits = np.asarray(bits)
    if bits.size % QPSK_BITS_PER_SYMBOL != 0:
        raise ValueError(
            f"QPSK needs an even number of bits, got {bits.size}"
        )
    pairs = bits.reshape(-1, 2).astype(np.float64)
    i_part = 1.0 - 2.0 * pairs[:, 0]   # 0 -> +1, 1 -> -1
    q_part = 1.0 - 2.0 * pairs[:, 1]
    return (i_part + 1j * q_part) / np.sqrt(2.0)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    Decide which bits were sent, from (noisy) QPSK symbols.

    The nearest QPSK point is simply the one in the same quadrant, so:
    real part < 0 -> first bit was 1; imaginary part < 0 -> second bit was 1.
    """
    symbols = np.asarray(symbols)
    bits = np.empty(symbols.size * 2, dtype=np.uint8)
    bits[0::2] = (symbols.real < 0).astype(np.uint8)
    bits[1::2] = (symbols.imag < 0).astype(np.uint8)
    return bits


# ---------------------------------------------------------------------------
# 3. AWGN channel (additive white Gaussian noise)
# ---------------------------------------------------------------------------
def ebn0_db_to_noise_std(ebn0_db: float, bits_per_symbol: int) -> float:
    """
    Convert Eb/N0 in dB to the noise standard deviation PER DIMENSION
    (real and imaginary parts each get this much noise).

    With Es = 1:  Eb = 1 / bits_per_symbol
                  N0 = Eb / (Eb/N0 as a plain ratio)
    Complex noise of total power N0 is split equally between the real and
    imaginary parts, so each part has variance N0 / 2.
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    eb = 1.0 / bits_per_symbol
    n0 = eb / ebn0_linear
    return float(np.sqrt(n0 / 2.0))


def add_awgn(
    symbols: np.ndarray,
    ebn0_db: float,
    bits_per_symbol: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex Gaussian noise at the given Eb/N0 (dB)."""
    std = ebn0_db_to_noise_std(ebn0_db, bits_per_symbol)
    noise = std * (
        rng.standard_normal(symbols.size) + 1j * rng.standard_normal(symbols.size)
    )
    return symbols + noise


# ---------------------------------------------------------------------------
# 4. Measuring errors
# ---------------------------------------------------------------------------
def count_bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> int:
    """How many positions differ between sent and received bits."""
    if tx_bits.shape != rx_bits.shape:
        raise ValueError(
            f"bit arrays differ in shape: {tx_bits.shape} vs {rx_bits.shape}"
        )
    return int(np.count_nonzero(tx_bits != rx_bits))


def theoretical_ber_qpsk(ebn0_db) -> np.ndarray:
    """
    Textbook BER of Gray-coded QPSK over AWGN:

        BER = 0.5 * erfc( sqrt(Eb/N0) )

    (Same as BPSK, because QPSK is two independent BPSK streams on I and Q.)
    """
    ebn0_linear = 10.0 ** (np.asarray(ebn0_db, dtype=np.float64) / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0_linear))


# ---------------------------------------------------------------------------
# 5. One full simulated transmission
# ---------------------------------------------------------------------------
def simulate_qpsk_awgn(ebn0_db: float, n_bits: int, seed: int) -> dict:
    """
    Send n_bits through QPSK + AWGN at ebn0_db and measure the BER.

    Returns a dict so later phases (the test runner) can log every field.
    """
    rng = np.random.default_rng(seed)
    tx_bits = generate_bits(n_bits, rng)
    tx_symbols = qpsk_modulate(tx_bits)
    rx_symbols = add_awgn(tx_symbols, ebn0_db, QPSK_BITS_PER_SYMBOL, rng)
    rx_bits = qpsk_demodulate(rx_symbols)
    errors = count_bit_errors(tx_bits, rx_bits)
    return {
        "modulation": "QPSK",
        "channel": "awgn",
        "ebn0_db": float(ebn0_db),
        "n_bits": int(n_bits),
        "bit_errors": errors,
        "ber": errors / n_bits,
        "ber_theory": float(theoretical_ber_qpsk(ebn0_db)),
    }


# ---------------------------------------------------------------------------
# Run this file directly to see simulated vs theoretical BER side by side:
#     python -m sim.channel
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import SEED

    print("QPSK over AWGN: simulated vs theoretical BER (1,000,000 bits each)")
    print(f"{'Eb/N0 (dB)':>10} | {'errors':>7} | {'simulated':>10} | "
          f"{'theory':>10} | {'diff %':>7}")
    print("-" * 56)
    for ebn0 in [0, 2, 4, 6, 8]:
        r = simulate_qpsk_awgn(ebn0, n_bits=1_000_000, seed=SEED + ebn0)
        diff = 100.0 * (r["ber"] - r["ber_theory"]) / r["ber_theory"]
        print(f"{ebn0:>10} | {r['bit_errors']:>7} | {r['ber']:>10.3e} | "
              f"{r['ber_theory']:>10.3e} | {diff:>+6.1f}%")