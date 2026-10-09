# it proves the simulator is correct  These tests don't prove that the entire simulator is correct. 
# They check specific properties of the simulator against mathematical expectations and known behaviors. 
# If all the tests pass, you gain evidence that the implementation is behaving correctly under the conditions tested.

"""
tests/test_channel.py - checks that the radio simulator is correct.

The most important test is test_ber_matches_theory: we only trust the lab's
data after the simulated QPSK BER agrees with the textbook formula.
"""

import numpy as np
import pytest

from sim.channel import (
    add_awgn,
    count_bit_errors,
    generate_bits,
    qpsk_demodulate,
    qpsk_modulate,
    simulate_qpsk_awgn,
    theoretical_ber_qpsk,
)


def test_symbols_have_unit_energy():
    rng = np.random.default_rng(0)
    symbols = qpsk_modulate(generate_bits(10_000, rng))
    # every QPSK symbol has |s|^2 = 1, so the average is exactly 1
    assert np.allclose(np.abs(symbols) ** 2, 1.0)


def test_gray_mapping_points():
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    expected = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    assert np.allclose(qpsk_modulate(bits), expected)


def test_roundtrip_without_noise_is_perfect():
    rng = np.random.default_rng(1)
    bits = generate_bits(20_000, rng)
    assert count_bit_errors(bits, qpsk_demodulate(qpsk_modulate(bits))) == 0


def test_odd_number_of_bits_is_rejected():
    with pytest.raises(ValueError, match="even number of bits"):
        qpsk_modulate(np.array([0, 1, 1], dtype=np.uint8))


def test_noise_power_matches_ebn0():
    # At Eb/N0 = 0 dB with Es = 1 and 2 bits/symbol: N0 = 0.5,
    # so the measured complex noise power should be close to 0.5.
    rng = np.random.default_rng(2)
    clean = np.zeros(200_000, dtype=complex)
    noise = add_awgn(clean, ebn0_db=0.0, bits_per_symbol=2, rng=rng)
    assert np.mean(np.abs(noise) ** 2) == pytest.approx(0.5, rel=0.02)


@pytest.mark.parametrize("ebn0_db", [0, 2, 4, 6])
def test_ber_matches_theory(ebn0_db):
    # 1,000,000 bits gives over 2,000 errors even at 6 dB,
    # so the simulated BER should land within 10% of theory.
    result = simulate_qpsk_awgn(ebn0_db, n_bits=1_000_000, seed=100 + ebn0_db)
    theory = theoretical_ber_qpsk(ebn0_db)
    assert result["ber"] == pytest.approx(theory, rel=0.10)


def test_more_noise_means_more_errors():
    bers = [simulate_qpsk_awgn(e, 200_000, seed=7)["ber"] for e in (0, 4, 8)]
    assert bers[0] > bers[1] > bers[2]


def test_same_seed_gives_same_result():
    a = simulate_qpsk_awgn(3.0, 100_000, seed=42)
    b = simulate_qpsk_awgn(3.0, 100_000, seed=42)
    assert a == b