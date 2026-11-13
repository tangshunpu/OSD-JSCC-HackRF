"""CPU tests of the 10 MHz HackRF OFDM link PHY (no radio, no model, no downloads).

    python3 -m unittest experiments/deploy/test_ofdm_phy.py

The frozen 1 MS/s PHY of the 2026-11-06 measurements has its own tests in
``test_ofdm_phy_1m.py``.
"""

from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ofdm_phy as phy  # noqa: E402

LAYERS = [(0, 2), (2, 4), (4, 8), (8, 16)]  # CR 96 prefix of the journal model
MULTIPATH = [1.0] + [0] * 15 + [0.22 + 0.13j] + [0] * 7 + [-0.1j]  # echoes at 0.8 and 1.2 us (CP 1.6 us)


def unit_power_symbols(height=512, width=768, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((16, height // 16, width // 16)) * np.sqrt(0.5)
    for c0, c1 in LAYERS:  # per-layer E[x^2] = 1/2, as WirelessChannel._normalize
        x[c0:c1] /= np.sqrt(2 * np.mean(x[c0:c1] ** 2))
    return x.astype(np.float16)


def evm(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)))


class ConvolutionalCodeTest(unittest.TestCase):
    def test_noiseless_round_trip_and_termination(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            bits = rng.integers(0, 2, phy.HEADER_BITS)
            coded = phy.conv_encode(bits)
            self.assertEqual(len(coded), phy.CODED_BITS)
            decoded = phy.viterbi(2.0 * coded - 1.0)
            np.testing.assert_array_equal(decoded[:phy.HEADER_BITS], bits)
            np.testing.assert_array_equal(decoded[phy.HEADER_BITS:], 0)

    def test_free_distance_is_ten(self):
        # weight of the impulse response = d_free of the 133/171 code
        self.assertEqual(int(phy.conv_encode(np.r_[1, np.zeros(20, int)]).sum()), 10)

    def test_coding_gain_over_repetition(self):
        """Same energy per header bit: K=7 code vs the old two-copy repetition, Eb/N0 = 4 dB."""
        rng = np.random.default_rng(2)
        ebn0 = 10 ** 0.4
        sigma = np.sqrt(phy.HEADER_SYMBOLS / phy.HEADER_BITS / (2 * ebn0))  # per real symbol, unit energy
        coded_errors = repeat_errors = 0
        trials = 300
        for _ in range(trials):
            bits = rng.integers(0, 2, phy.HEADER_BITS)
            tx = 2.0 * phy.conv_encode(bits) - 1.0
            out = phy.viterbi(tx + sigma * rng.standard_normal(len(tx)))[:phy.HEADER_BITS]
            coded_errors += bool(np.any(out != bits))
            rep = np.tile(2.0 * bits - 1.0, 2) + sigma * rng.standard_normal(2 * len(bits))
            repeat_errors += bool(np.any(((rep[:len(bits)] + rep[len(bits):]) > 0) != bits))
        self.assertLess(coded_errors, trials * 0.05)
        self.assertGreater(repeat_errors, trials * 0.5)

    def test_header_round_trip_and_rejection(self):
        symbols = phy.header_symbols(seq=40001, packet=5, total=16)
        self.assertEqual(symbols.shape, (phy.HEADER_ROWS, 48))
        self.assertEqual(phy.parse_header(symbols, 16, 40001), (5, 40001))
        self.assertIsNone(phy.parse_header(symbols, 16, 40002))   # other image
        self.assertIsNone(phy.parse_header(symbols, 8, 40001))    # other packet count
        self.assertIsNone(phy.parse_header(-symbols, 16, 40001))  # garbage fails the CRC


class OFDMPhyTest(unittest.TestCase):
    def setUp(self):
        self.symbols = unit_power_symbols()
        self.perm = phy.permutation(phy.DEFAULT_PERMUTATION_SEED)
        self.scramble = phy.DEFAULT_SCRAMBLE_SEED
        self.rows = phy.air_rows(self.symbols, LAYERS, phy.DEFAULT_PERMUTATION_SEED, self.scramble)

    def receive(self, iq, seq, total=8):
        return phy.recover(phy.to_baseband(iq), total, seq)

    def symbols_out(self, packets):
        return phy.rows_to_symbols(phy.assemble(packets, 8, self.perm, self.scramble), LAYERS, 512, 768)

    def test_mapping_round_trip_both_orientations(self):
        for height, width in ((512, 768), (768, 512)):
            symbols = unit_power_symbols(height, width, seed=1)
            rows = phy.symbols_to_rows(symbols, LAYERS)
            self.assertEqual(rows.shape, (256, 48))
            back = phy.rows_to_symbols(phy.permute_rows(phy.permute_rows(rows, self.perm), self.perm, inverse=True),
                                       LAYERS, height, width)
            np.testing.assert_array_equal(back.astype(np.float16), symbols)

    def test_waveform_timing(self):
        iq, info = phy.modulate(self.rows, seq=1)
        self.assertEqual(info["packets"], 8)
        self.assertAlmostEqual(phy.PACKET_LEN / phy.BASE_RATE, 320e-6)
        self.assertLess(info["seconds"], 0.004)  # 8 packets + guards, no settling time in stream mode
        starts = np.array(info["packet_starts"])
        np.testing.assert_array_equal(np.diff(starts), (phy.PACKET_LEN + phy.GUARD) * phy.UP)
        _, burst = phy.modulate(self.rows, seq=1, lead=int(phy.SETTLE_SECONDS * phy.BASE_RATE))
        self.assertAlmostEqual(burst["seconds"] - info["seconds"], phy.SETTLE_SECONDS, places=6)

    def test_native_kernel_matches_numpy(self):
        """The C++ modulator gives the numpy path's I/Q samples and metadata (needs a compiler)."""
        if phy.native_kernel() is None:
            self.skipTest("no C++ compiler")
        rng = np.random.default_rng(9)
        smooth = phy.payload_to_rows(np.repeat(rng.standard_normal(2 * phy.ROWS * 96 // 8), 8), "real")
        for rows, ids, lead in ((self.rows, None, 0), (self.rows, [5, 2], 1000), (smooth, None, 0)):
            a, ia = phy.modulate(rows, 77, ids, lead=lead, native=True)
            b, ib = phy.modulate(rows, 77, ids, lead=lead, native=False)
            np.testing.assert_array_equal(a, b)
            for key, value in ib.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(ia[key], value, delta=1e-12 * abs(value))
                else:
                    self.assertEqual(ia[key], value)

    def test_native_receiver_matches_numpy(self):
        """The C++ receiver finds the same packets with the same report as numpy (needs a compiler)."""
        if phy.native_kernel() is None:
            self.skipTest("no C++ compiler")
        iq, _ = phy.modulate(self.rows, seq=31, packet_ids=[0, 3, 4, 7])
        rx = phy.simulate_channel(iq, cfo_hz=-6100.0, delay=377, phase=2.1, taps=MULTIPATH, noise_counts=4.0,
                                  lead_seconds=0.0002)
        z = phy.to_baseband(rx[: len(rx) - 9000])  # cuts into the last packet
        for tracking, method in (("hybrid", "sure"), ("packet", "smooth"), ("symbol", "ls")):
            a, ra = phy.recover(z, 8, 31, tracking, method, native=True)
            b, rb = phy.recover(z, 8, 31, tracking, method, native=False)
            self.assertAlmostEqual(ra.pop("removed_dc"), rb.pop("removed_dc"), delta=1e-9)
            self.assertEqual(ra, rb)
            self.assertEqual(sorted(a), [0, 3, 4])
            for packet in a:
                np.testing.assert_allclose(a[packet][0], b[packet][0], rtol=0, atol=1e-9)
                for key, value in b[packet][1].items():
                    if isinstance(value, float):
                        self.assertAlmostEqual(a[packet][1][key], value, delta=1e-7 * max(1.0, abs(value)))
                    elif isinstance(value, list):
                        np.testing.assert_allclose(a[packet][1][key], value, rtol=1e-7)
                    else:
                        self.assertEqual(a[packet][1][key], value)

    def test_impaired_channel_recovers_all_packets(self):
        iq, info = phy.modulate(self.rows, seq=4242)
        self.assertLessEqual(info["normalized_peak"], phy.PEAK_LIMIT)
        rx = phy.simulate_channel(iq, cfo_hz=48_500.0, delay=1234, phase=0.73, noise_counts=0.3,
                                  taps=MULTIPATH, spur=2.0 + 0.5j)
        packets, report = self.receive(rx, 4242)
        self.assertEqual(report["decoded_packets"], 8, report)
        for item in (p[1] for p in packets.values()):
            self.assertAlmostEqual(item["total_cfo_hz"], 48_500.0, delta=200)
        self.assertLess(evm(self.symbols_out(packets), self.symbols.astype(np.float32)), 0.08)
        self.assertGreater(phy.snr_estimate([p[1] for p in packets.values()]), 20)

    def test_continuous_tx_lo_leakage_tone(self):
        """A TX LO tone as strong as the signal RMS, on before/between/after packets."""
        iq, _ = phy.modulate(self.rows, seq=77)
        tone = 12.0 * np.exp(0.4j)
        rx = phy.simulate_channel(iq, cfo_hz=-61_000.0, delay=999, noise_counts=1.0, tone=tone, seed=4)
        start = time.perf_counter()
        packets, report = self.receive(rx, 77)
        self.assertEqual(report["decoded_packets"], 8, report)
        self.assertLess(report["rejected_candidates"], 20)  # the tone alone does not trigger detection
        self.assertLess(evm(self.symbols_out(packets), self.symbols.astype(np.float32)), 0.12)
        idle = phy.simulate_channel(np.zeros((40_000, 2), np.int8), cfo_hz=-61_000.0, noise_counts=1.0,
                                    tone=tone, seed=5)
        found, report = self.receive(idle, None)
        self.assertEqual(found, {})
        self.assertLess(report["rejected_candidates"], 5)
        self.assertLess(time.perf_counter() - start, 5.0)

    def test_noise_and_header_snr_track_each_other(self):
        iq, _ = phy.modulate(self.rows, seq=7)
        rx = phy.simulate_channel(iq, cfo_hz=-3000.0, noise_counts=4.0, seed=3)
        packets, report = self.receive(rx, 7)
        self.assertEqual(report["decoded_packets"], 8, report)
        payload_snr = -20 * np.log10(evm(self.symbols_out(packets), self.symbols.astype(np.float32)))
        header_snr = phy.snr_estimate([p[1] for p in packets.values()])
        self.assertAlmostEqual(header_snr, payload_snr, delta=1.5)

    def test_probe_snr_matches_header_snr(self):
        iq, _ = phy.modulate(phy.probe_rows(31), seq=31)
        packets, _ = self.receive(phy.simulate_channel(iq, cfo_hz=9000.0, noise_counts=3.0, seed=9), 31, total=2)
        self.assertEqual(sorted(packets), [0, 1])
        self.assertAlmostEqual(phy.probe_snr(packets, 31), phy.snr_estimate([p[1] for p in packets.values()]),
                               delta=1.0)

    def test_snr_estimates_ignore_short_bursts(self):
        """Strong 8 us bursts on one header symbol of three packets move the median-based
        estimate by < 0.5 dB (the mean would drop by many dB)."""
        iq, info = phy.modulate(self.rows, seq=21)
        rx = phy.simulate_channel(iq, noise_counts=1.5, seed=2).astype(float)
        rng = np.random.default_rng(3)
        lead = int(0.005 * phy.RATE)
        for start in np.array(info["packet_starts"])[[0, 3, 6]]:  # header symbol 2 of three packets
            a = lead + start + (len(phy.TRAIN) + 2 * phy.SYMLEN) * phy.UP
            rx[a:a + phy.SYMLEN * phy.UP] += rng.normal(0, 12, (phy.SYMLEN * phy.UP, 2))
        hit = np.clip(np.rint(rx), -128, 127).astype(np.int8)
        clean, _ = self.receive(phy.simulate_channel(iq, noise_counts=1.5, seed=2), 21)
        packets, _ = self.receive(hit, 21)
        self.assertEqual(len(packets), 8)
        self.assertAlmostEqual(phy.snr_estimate([p[1] for p in packets.values()]),
                               phy.snr_estimate([p[1] for p in clean.values()]), delta=0.5)

    def test_coded_header_survives_low_snr(self):
        """Near the STF detection limit every detected packet's header still decodes."""
        iq, _ = phy.modulate(self.rows, seq=5)
        decoded = []
        for seed in range(4):
            rx = phy.simulate_channel(iq, cfo_hz=2000.0, noise_counts=17.0, seed=seed)  # about 1 dB
            packets, report = self.receive(rx, 5)
            decoded.append(report["decoded_packets"])
        self.assertGreaterEqual(np.mean(decoded), 6, decoded)

    def test_scrambling_bounds_papr_of_smooth_features(self):
        smooth = np.repeat(np.linspace(-1, 1, 16)[:, None, None], 32 * 48, axis=1).reshape(16, 32, 48)
        plain = phy.air_rows(smooth, LAYERS, scramble_seed=None)
        scrambled = phy.air_rows(smooth, LAYERS)
        self.assertGreater(phy.modulate(plain, 1)[1]["papr_active_db"], 16)
        info = phy.modulate(scrambled, 1)[1]
        self.assertLess(info["papr_active_db"], 14)
        self.assertFalse(info["rms_limited_by_peak"])
        packets, _ = phy.recover(phy.to_baseband(phy.simulate_channel(phy.modulate(scrambled, 1)[0])), 8, 1)
        out = phy.rows_to_symbols(phy.assemble(packets, 8, self.perm, self.scramble), LAYERS, 512, 768)
        self.assertLess(evm(out, smooth), 0.03)

    def test_generic_payload_round_trip(self):
        rng = np.random.default_rng(11)
        real = rng.standard_normal(24576).astype(np.float32)
        rows = phy.payload_to_rows(real, "real")
        self.assertEqual(rows.shape, (256, 48))
        np.testing.assert_allclose(phy.rows_to_payload(rows, "real", len(real)), real, rtol=1e-6)
        qam = (rng.choice([-3, -1, 1, 3], 11111) + 1j * rng.choice([-3, -1, 1, 3], 11111)) / np.sqrt(10)
        rows = phy.payload_to_rows(qam, "complex", filler_seed=5)
        self.assertEqual(rows.shape, (256, 48))
        self.assertAlmostEqual(float(np.mean(np.abs(rows) ** 2)), 1.0, delta=0.03)
        air = phy.permute_rows(rows, self.perm) * phy.scrambling(len(rows), self.scramble)
        iq, _ = phy.modulate(air, 3)
        packets, _ = phy.recover(phy.to_baseband(phy.simulate_channel(iq, cfo_hz=-7000.0, noise_counts=0.3)), 8, 3)
        out = phy.rows_to_payload(phy.assemble(packets, 8, self.perm, self.scramble), "complex", len(qam))
        self.assertLess(float(np.sqrt(np.mean(np.abs(out - qam) ** 2))), 0.06)

    def test_other_sequence_is_rejected(self):
        iq, _ = phy.modulate(self.rows, seq=100)
        rx = phy.simulate_channel(iq, cfo_hz=1000.0, noise_counts=0.3)
        packets, report = self.receive(rx, 101)
        self.assertEqual(report["decoded_packets"], 0)
        self.assertGreater(report["rejected_candidates"], 0)

    def test_packet_subset(self):
        iq, info = phy.modulate(self.rows, seq=9, packet_ids=[2, 5])
        self.assertEqual(info["packets"], 2)
        packets, report = self.receive(phy.simulate_channel(iq, noise_counts=0.3), 9)
        self.assertEqual(sorted(packets), [2, 5])
        self.assertEqual(report["missing_packet_ids"], [0, 1, 3, 4, 6, 7])
        rows = phy.assemble(packets, 8, self.perm, self.scramble)
        self.assertTrue(np.isnan(rows[:64]).all() and np.isfinite(rows[64:96]).all())

    def test_noise_only_has_no_packets(self):
        rng = np.random.default_rng(5)
        noise = np.clip(np.rint(rng.normal(0, 3, (800_000, 2))), -128, 127).astype(np.int8)
        packets, _ = self.receive(noise, None)
        self.assertEqual(packets, {})

    def test_channel_smoothing_keeps_delay_ramp(self):
        h = 3 * np.exp(1j * (0.4 + 2 * np.pi * 4 * phy.ACTIVE / phy.NFFT))  # flat channel, FFT window 4 samples early
        np.testing.assert_allclose(phy._smooth_channel(h, 7), h, rtol=1e-12)

    def test_sure_channel_estimate_follows_selective_channels(self):
        """Echo at 0.8 us: SURE beats the fixed 7-carrier average; flat: it matches it."""
        iq, _ = phy.modulate(self.rows, seq=12)
        result = {}
        for name, taps in (("flat", (1.0,)), ("echo", [1.0] + [0] * 15 + [0.5j])):
            z = phy.to_baseband(phy.simulate_channel(iq, cfo_hz=5000.0, noise_counts=0.5, taps=taps, seed=1))
            for method in ("smooth", "sure"):
                packets, _ = phy.recover(z, 8, 12, channel_method=method)
                result[name, method] = -20 * np.log10(evm(self.symbols_out(packets), self.symbols.astype(np.float32)))
        self.assertGreater(result["echo", "sure"], result["echo", "smooth"] + 5, result)
        self.assertGreater(result["flat", "sure"], result["flat", "smooth"] - 1, result)

    def test_detection_signal_keeps_stf(self):
        z = np.r_[np.zeros(64), phy.STF, np.zeros(64)] + 0.0j
        out = phy.detection_signal(z)
        np.testing.assert_allclose(out[64 + 16:64 + 160 - 16], phy.STF[16:-16], atol=1e-12)


class PaddedPacketTest(unittest.TestCase):
    def test_partial_last_packet_round_trip(self):
        """CR 768 at 768x1280: one 2-channel layer = 80 rows, filled up to 3 packets of 32."""
        layers, height, width = [(0, 2)], 768, 1280
        symbols = unit_power_symbols(height, width, seed=7)[:2]
        rows = phy.air_rows(symbols, layers)
        self.assertEqual(len(rows), 96)
        iq, info = phy.modulate(rows, seq=9)
        self.assertEqual(info["packets"], 3)
        packets, _ = phy.recover(phy.to_baseband(phy.simulate_channel(iq, noise_counts=0.5, seed=1)), 3, 9)
        self.assertEqual(len(packets), 3)
        back = phy.rows_to_symbols(phy.assemble(packets, 3, phy.permutation(phy.DEFAULT_PERMUTATION_SEED),
                                                phy.DEFAULT_SCRAMBLE_SEED), layers, height, width)
        self.assertLess(float(np.mean((back - symbols.astype(np.float32)) ** 2)), 0.05)


class InferredPacketTest(unittest.TestCase):
    """recover(infer_ids=True): header-CRC failures identified by their slot on the burst grid."""

    def test_rules(self):
        total, seq, origin = 8, 4321, 10_000
        starts = origin + np.arange(total) * phy.PACKET_SPACING
        bits = [phy.header_bits(seq, k, total) for k in range(total)]
        parsed = [(k, seq) for k in range(total)]
        bad = bits[3].copy()
        bad[[5, 17, 40, 41, 66]] ^= 1                       # 5 bit errors: CRC fails, 94 % agree
        self.assertIsNone(phy._check_header(np.r_[bad, np.zeros(6, np.uint8)], total, seq))
        cases = {
            "few bit errors": (starts[3], bad, {3}),
            "off the grid": (starts[3] + 9, bad, set()),
            "other transmission": (starts[3], phy.header_bits(seq + 1, 3, total), set()),
            "noise": (starts[3], np.random.default_rng(0).integers(0, 2, 80).astype(np.uint8), set()),
        }
        for name, (start, header, expected) in cases.items():
            p, s, b = list(parsed), starts.astype(float).copy(), list(bits)
            p[3], s[3], b[3] = None, start, header
            got = phy._infer_packet_ids(p, np.ones(total, bool), s, np.ones(total), b, total, seq)
            self.assertEqual({k for k, _ in got.values()}, expected, name)

    def test_low_snr_burst(self):
        """Near the detection limit: more packets, and each inferred one carries its own slot's payload."""
        rows = phy.air_rows(unit_power_symbols(), LAYERS)
        iq, _ = phy.modulate(rows, seq=5)
        gained = 0
        for native in (True, False) if phy.native_kernel() is not None else (False,):
            for seed in range(8):  # noise 24 counts: about -2 dB, a few headers fail their CRC
                z = phy.to_baseband(phy.simulate_channel(iq, cfo_hz=2000.0, noise_counts=24.0, seed=seed))
                plain, _ = phy.recover(z, 8, 5, native=native)
                more, report = phy.recover(z, 8, 5, native=native, infer_ids=True)
                self.assertTrue(set(plain) <= set(more))
                self.assertEqual(report["inferred_packets"], len(more) - len(plain))
                gained += len(more) - len(plain)
                errors = {k: np.mean(np.abs(v[0] - rows[k * phy.ROWS:(k + 1) * phy.ROWS]) ** 2) for k, v in more.items()}
                for k, (_, info) in more.items():
                    if info.get("header_crc") is False:  # right slot: error like a valid packet, not ~2
                        self.assertLess(errors[k], 1.5, (seed, k, errors))
                        self.assertIn("header_row_errors", info)
        self.assertGreater(gained, 0)


if __name__ == "__main__":
    unittest.main()
