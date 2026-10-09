# # it proves the simulator is correct  These tests don't prove that the entire simulator is correct. 
# # They check specific properties of the simulator against mathematical expectations and known behaviors. 
# # If all the tests pass, you gain evidence that the implementation is behaving correctly under the conditions tested.

"""
tests/test_channel.py - checks that the radio simulator is correct.

The most important tests compare simulated BER with textbook theory:
we only trust the lab's data after the simulator agrees with theory.
"""

import numpy as np
import pytest

from sim.channel import (
    add_awgn,
    count_bit_errors,
    generate_bits,
    qam16_demodulate,
    qam16_modulate,
    qpsk_demodulate,
    qpsk_modulate,
    rayleigh_gains,
    simulate_link,
    simulate_qpsk_awgn,
    theoretical_ber_16qam,
    theoretical_ber_qpsk,
    theoretical_ber_qpsk_rayleigh,
)


# ===========================================================================
# Phase 1 tests: QPSK over AWGN
# ===========================================================================
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


# ===========================================================================
# Phase 2 tests: 16-QAM, Rayleigh fading, equaliser, impairment hooks
# ===========================================================================
def test_16qam_average_energy_is_one():
    rng = np.random.default_rng(3)
    symbols = qam16_modulate(generate_bits(400_000, rng))
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0, rel=0.01)


def test_16qam_gray_levels():
    # I bits 00,01,11,10 -> -3,-1,+1,+3 (Q bits fixed at 00 -> -3)
    bits = np.array([0, 0, 0, 0,  0, 1, 0, 0,  1, 1, 0, 0,  1, 0, 0, 0],
                    dtype=np.uint8)
    expected = (np.array([-3, -1, 1, 3]) - 3j) / np.sqrt(10)
    assert np.allclose(qam16_modulate(bits), expected)


def test_16qam_roundtrip_without_noise_is_perfect():
    rng = np.random.default_rng(4)
    bits = generate_bits(40_000, rng)
    assert count_bit_errors(bits, qam16_demodulate(qam16_modulate(bits))) == 0


def test_16qam_rejects_wrong_bit_count():
    with pytest.raises(ValueError, match="multiple of 4"):
        qam16_modulate(np.zeros(6, dtype=np.uint8))


@pytest.mark.parametrize("ebn0_db", [6, 8, 10])
def test_16qam_ber_matches_theory(ebn0_db):
    r = simulate_link("16QAM", "awgn", ebn0_db, 1_000_000, seed=200 + ebn0_db)
    assert r["ber"] == pytest.approx(theoretical_ber_16qam(ebn0_db), rel=0.10)


def test_rayleigh_average_power_is_one():
    h = rayleigh_gains(400_000, np.random.default_rng(5))
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("ebn0_db", [5, 10, 15])
def test_qpsk_rayleigh_ber_matches_theory(ebn0_db):
    r = simulate_link("QPSK", "rayleigh", ebn0_db, 1_000_000, seed=300 + ebn0_db)
    assert r["ber"] == pytest.approx(
        theoretical_ber_qpsk_rayleigh(ebn0_db), rel=0.10
    )


def test_fading_is_worse_than_awgn():
    awgn = simulate_link("QPSK", "awgn", 8, 200_000, seed=9)["ber"]
    fading = simulate_link("QPSK", "rayleigh", 8, 200_000, seed=9)["ber"]
    assert fading > 10 * awgn


def test_16qam_needs_more_ebn0_than_qpsk():
    qpsk = simulate_link("QPSK", "awgn", 8, 200_000, seed=10)["ber"]
    qam = simulate_link("16QAM", "awgn", 8, 200_000, seed=10)["ber"]
    assert qam > qpsk


# ---- impairment hooks: each bug must break what it should, and only that ----
def test_equalizer_off_breaks_fading_only():
    off = {"equalizer_enabled": False}
    fading = simulate_link("QPSK", "rayleigh", 15, 200_000, 11, off)["ber"]
    awgn = simulate_link("QPSK", "awgn", 15, 200_000, 11, off)["ber"]
    assert fading > 0.3          # close to guessing (0.5)
    assert awgn < 1e-3           # equaliser is not used on AWGN


def test_tx_scale_bug_hurts_16qam_not_qpsk():
    bug = {"tx_scale": 0.8}
    qam_bug = simulate_link("16QAM", "awgn", 12, 200_000, 12, bug)["ber"]
    qam_ok = simulate_link("16QAM", "awgn", 12, 200_000, 12)["ber"]
    qpsk_bug = simulate_link("QPSK", "awgn", 12, 200_000, 12, bug)["ber"]
    assert qam_bug > 10 * qam_ok
    assert qpsk_bug < 1e-4       # QPSK only looks at signs, so scale is harmless


def test_phase_offset_hurts_16qam_more_than_qpsk():
    bug = {"phase_offset_deg": 20}
    qam = simulate_link("16QAM", "awgn", 12, 200_000, 13, bug)["ber"]
    qpsk = simulate_link("QPSK", "awgn", 12, 200_000, 13, bug)["ber"]
    assert qam > 50 * max(qpsk, 1e-6)


def test_snr_offset_equals_lower_ebn0():
    bug = simulate_link("QPSK", "awgn", 8, 200_000, 14, {"snr_offset_db": -3})
    plain = simulate_link("QPSK", "awgn", 5, 200_000, 14)
    assert bug["ber"] == plain["ber"]


def test_unknown_impairment_is_rejected():
    with pytest.raises(ValueError, match="unknown impairment"):
        simulate_link("QPSK", "awgn", 8, 1_000, 0, {"antenna_on_fire": True})


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="unknown channel"):
        simulate_link("QPSK", "underwater", 8, 1_000, 0)