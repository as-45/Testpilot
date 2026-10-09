"""
sim/channel.py - the radio physics of TestPilot.

Phase 1: QPSK over AWGN, validated against theory.
Phase 2: 16-QAM, Rayleigh fading, a zero-forcing equaliser, and
         "impairment" hooks that Phase 3 uses to inject bugs.

The chain this file simulates:

    bits -> modulate -> [fading h] -> + noise -> [equaliser] -> demodulate -> bits
                                                                            |
                                  compare with the bits we sent  <----------+
                                  -> BER (bit error rate)

Conventions (fixed for the whole project):
  * Symbols are scaled so their average energy Es = 1.
  * Noise level is given as Eb/N0 in dB (energy per bit / noise density).
  * QPSK carries 2 bits per symbol, 16-QAM carries 4.
"""

import numpy as np
from scipy.special import erfc

QPSK_BITS_PER_SYMBOL = 2
QAM16_BITS_PER_SYMBOL = 4
BITS_PER_SYMBOL = {"QPSK": QPSK_BITS_PER_SYMBOL, "16QAM": QAM16_BITS_PER_SYMBOL}
CHANNELS = ("awgn", "rayleigh")


# ---------------------------------------------------------------------------
# 1. Bits
# ---------------------------------------------------------------------------
def generate_bits(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Return n_bits random 0/1 values as a uint8 array."""
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 2a. QPSK modulation / demodulation  (unchanged from Phase 1)
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
# 2b. 16-QAM modulation / demodulation  (new in Phase 2)
# ---------------------------------------------------------------------------
# Each axis (I and Q) carries 2 bits on 4 levels.  Gray order means
# neighbouring levels differ by exactly one bit:
#     bits 00 -> -3,  01 -> -1,  11 -> +1,  10 -> +3
_QAM16_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])
_QAM16_BITS_FROM_INDEX = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
# average energy of the 16 points (+-1, +-3 on each axis) is 10 -> divide by sqrt(10)
_QAM16_NORM = np.sqrt(10.0)


def _bits_to_level(b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Map Gray bit pairs to the levels -3, -1, +1, +3 (vectorised)."""
    # index = position in _QAM16_LEVELS:  00->0, 01->1, 11->2, 10->3
    index = np.where(b0 == 0, b1, 3 - b1)
    return _QAM16_LEVELS[index]


def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Turn bits into 16-QAM symbols (Gray mapping), 4 bits per symbol.

    Bits 1-2 choose the I level, bits 3-4 choose the Q level.
    Divided by sqrt(10) so the average symbol energy is 1, same as QPSK.
    """
    bits = np.asarray(bits)
    if bits.size % QAM16_BITS_PER_SYMBOL != 0:
        raise ValueError(
            f"16-QAM needs a multiple of 4 bits, got {bits.size}"
        )
    quads = bits.reshape(-1, 4).astype(np.int64)
    i_level = _bits_to_level(quads[:, 0], quads[:, 1])
    q_level = _bits_to_level(quads[:, 2], quads[:, 3])
    return (i_level + 1j * q_level) / _QAM16_NORM


def _nearest_level_index(values: np.ndarray) -> np.ndarray:
    """For each value, the index (0..3) of the nearest level in -3,-1,+1,+3."""
    # decision boundaries sit halfway between levels: -2, 0, +2
    return np.digitize(values, [-2.0, 0.0, 2.0])


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Decide the nearest 16-QAM point on each axis and turn it back into bits."""
    symbols = np.asarray(symbols) * _QAM16_NORM   # undo the normalisation
    i_idx = _nearest_level_index(symbols.real)
    q_idx = _nearest_level_index(symbols.imag)
    bits = np.empty((symbols.size, 4), dtype=np.uint8)
    bits[:, 0:2] = _QAM16_BITS_FROM_INDEX[i_idx]
    bits[:, 2:4] = _QAM16_BITS_FROM_INDEX[q_idx]
    return bits.reshape(-1)


# ---------------------------------------------------------------------------
# 2c. Pick the right modulator by name
# ---------------------------------------------------------------------------
def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    if modulation == "QPSK":
        return qpsk_modulate(bits)
    if modulation == "16QAM":
        return qam16_modulate(bits)
    raise ValueError(f"unknown modulation {modulation!r}; use 'QPSK' or '16QAM'")


def demodulate(symbols: np.ndarray, modulation: str) -> np.ndarray:
    if modulation == "QPSK":
        return qpsk_demodulate(symbols)
    if modulation == "16QAM":
        return qam16_demodulate(symbols)
    raise ValueError(f"unknown modulation {modulation!r}; use 'QPSK' or '16QAM'")


# ---------------------------------------------------------------------------
# 3. Channels: AWGN and Rayleigh fading
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


def rayleigh_gains(n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """
    One random complex channel gain h per symbol (fast Rayleigh fading).

    h = (a + jb) / sqrt(2) with a, b standard normal, so the AVERAGE power
    |h|^2 is 1, but each symbol is randomly boosted, weakened and rotated -
    like a phone moving through a city.
    """
    return (
        rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)
    ) / np.sqrt(2.0)


def zero_forcing_equalize(received: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Undo the fading: divide each received symbol by its channel gain.

    We assume the receiver knows h exactly ("perfect channel estimate"),
    which keeps the simulator simple and lets us compare with theory.
    """
    return received / h


# ---------------------------------------------------------------------------
# 4. Measuring errors and theory curves
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


def theoretical_ber_16qam(ebn0_db) -> np.ndarray:
    """
    Standard approximation for Gray-coded 16-QAM over AWGN:

        BER ~= (3/8) * erfc( sqrt( (2/5) * Eb/N0 ) )

    It counts only mistakes to the nearest neighbour, so it is very close
    at moderate/high Eb/N0 and slightly off at very low Eb/N0.
    """
    ebn0_linear = 10.0 ** (np.asarray(ebn0_db, dtype=np.float64) / 10.0)
    return (3.0 / 8.0) * erfc(np.sqrt(0.4 * ebn0_linear))


def theoretical_ber_qpsk_rayleigh(ebn0_db) -> np.ndarray:
    """
    Textbook BER of QPSK over Rayleigh fading with perfect equalisation:

        BER = 0.5 * ( 1 - sqrt( g / (1 + g) ) ),   g = Eb/N0 (plain ratio)

    Much worse than AWGN at the same Eb/N0: deep fades cause most errors.
    """
    g = 10.0 ** (np.asarray(ebn0_db, dtype=np.float64) / 10.0)
    return 0.5 * (1.0 - np.sqrt(g / (1.0 + g)))


def theoretical_ber(modulation: str, channel: str, ebn0_db):
    """Theory curve for a (modulation, channel) pair, or None if we have none."""
    if modulation == "QPSK" and channel == "awgn":
        return theoretical_ber_qpsk(ebn0_db)
    if modulation == "16QAM" and channel == "awgn":
        return theoretical_ber_16qam(ebn0_db)
    if modulation == "QPSK" and channel == "rayleigh":
        return theoretical_ber_qpsk_rayleigh(ebn0_db)
    return None  # 16-QAM over Rayleigh: no simple closed form, we skip it


# ---------------------------------------------------------------------------
# 5. Impairments: the hooks Phase 3 uses to inject bugs
# ---------------------------------------------------------------------------
# A healthy device uses all the defaults below.  A buggy component changes
# one of them.  Each one models a real kind of modem bug:
#   snr_offset_db     rf_frontend: real signal weaker than the config says
#   tx_scale          mapper: constellation drawn too small (receiver still
#                     expects the normal size)
#   phase_offset_deg  sync: every symbol rotated by a fixed angle
#   equalizer_enabled equalizer: fading is not corrected when False
DEFAULT_IMPAIRMENTS = {
    "snr_offset_db": 0.0,
    "tx_scale": 1.0,
    "phase_offset_deg": 0.0,
    "equalizer_enabled": True,
}


def _check_impairments(impairments: dict | None) -> dict:
    merged = dict(DEFAULT_IMPAIRMENTS)
    if impairments:
        unknown = set(impairments) - set(DEFAULT_IMPAIRMENTS)
        if unknown:
            raise ValueError(f"unknown impairment(s): {sorted(unknown)}")
        merged.update(impairments)
    return merged


# ---------------------------------------------------------------------------
# 6. One full simulated transmission
# ---------------------------------------------------------------------------
def simulate_link(
    modulation: str,
    channel: str,
    ebn0_db: float,
    n_bits: int,
    seed: int,
    impairments: dict | None = None,
) -> dict:
    """
    Send n_bits through the chosen modulation and channel and measure BER.

    modulation:  "QPSK" or "16QAM"
    channel:     "awgn" or "rayleigh"
    impairments: optional dict of bugs (see DEFAULT_IMPAIRMENTS)

    Returns a dict so the test runner can log every field.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; use one of {CHANNELS}")
    if modulation not in BITS_PER_SYMBOL:
        raise ValueError(f"unknown modulation {modulation!r}; use 'QPSK' or '16QAM'")
    imp = _check_impairments(impairments)
    k = BITS_PER_SYMBOL[modulation]

    rng = np.random.default_rng(seed)
    tx_bits = generate_bits(n_bits, rng)
    tx_symbols = modulate(tx_bits, modulation)

    # --- transmitter-side impairments (mapper, sync) ---
    tx_symbols = tx_symbols * imp["tx_scale"]
    if imp["phase_offset_deg"]:
        tx_symbols = tx_symbols * np.exp(1j * np.deg2rad(imp["phase_offset_deg"]))

    # --- channel ---
    if channel == "rayleigh":
        h = rayleigh_gains(tx_symbols.size, rng)
        faded = tx_symbols * h
    else:
        h = None
        faded = tx_symbols
    effective_ebn0 = ebn0_db + imp["snr_offset_db"]       # rf_frontend bug
    rx_symbols = add_awgn(faded, effective_ebn0, k, rng)

    # --- receiver ---
    if channel == "rayleigh" and imp["equalizer_enabled"]:
        rx_symbols = zero_forcing_equalize(rx_symbols, h)
    rx_bits = demodulate(rx_symbols, modulation)

    errors = count_bit_errors(tx_bits, rx_bits)
    theory = theoretical_ber(modulation, channel, ebn0_db)
    return {
        "modulation": modulation,
        "channel": channel,
        "ebn0_db": float(ebn0_db),
        "n_bits": int(n_bits),
        "bit_errors": errors,
        "ber": errors / n_bits,
        "ber_theory": None if theory is None else float(theory),
    }


def simulate_qpsk_awgn(ebn0_db: float, n_bits: int, seed: int) -> dict:
    """Phase 1 helper, kept so earlier code and tests still work unchanged."""
    return simulate_link("QPSK", "awgn", ebn0_db, n_bits, seed)


# ---------------------------------------------------------------------------
# Run this file directly to see simulated vs theoretical BER side by side:
#     python -m sim.channel
# ---------------------------------------------------------------------------
def _print_table(modulation: str, channel: str, points, seed: int) -> None:
    print(f"\n{modulation} over {channel.upper()}: simulated vs theory "
          f"(1,000,000 bits each)")
    print(f"{'Eb/N0 (dB)':>10} | {'errors':>7} | {'simulated':>10} | "
          f"{'theory':>10} | {'diff %':>7}")
    print("-" * 56)
    for ebn0 in points:
        r = simulate_link(modulation, channel, ebn0, 1_000_000, seed + ebn0)
        diff = 100.0 * (r["ber"] - r["ber_theory"]) / r["ber_theory"]
        print(f"{ebn0:>10} | {r['bit_errors']:>7} | {r['ber']:>10.3e} | "
              f"{r['ber_theory']:>10.3e} | {diff:>+6.1f}%")


if __name__ == "__main__":
    from config import SEED

    _print_table("QPSK", "awgn", [0, 2, 4, 6, 8], SEED)
    _print_table("16QAM", "awgn", [4, 6, 8, 10, 12], SEED)
    _print_table("QPSK", "rayleigh", [0, 5, 10, 15, 20], SEED)