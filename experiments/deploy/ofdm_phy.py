"""OFDM physical layer that carries OSD-JSCC-P channel symbols between two HackRF One radios.

10 MHz version (2026-11-07) of the DeepStream-v2-style OFDM (arXiv:2509.05971v2)
first ported from ``../osd_jscc_hacker/jscc_ofdm.py``: 64-point FFT, 48 data +
4 pilot + 12 null carriers, CP 16, 10 MS/s baseband (156.25 kHz carrier
spacing, 8 us symbols, about 8.1 MHz occupied), HackRF at 20 MS/s with the LO
on the DC null carrier as in IEEE 802.11. Each packet is STF + LTF training, a
convolutionally coded BPSK header with CRC32, then 32 OFDM payload symbols. The
receiver does STF detection and coarse CFO, LTF timing, fine CFO and LS channel
estimation, ZF equalization and pilot phase tracking. It never reads the
transmitted payload. The 1 MS/s PHY of the 2026-11-06 measurements is frozen
in ``ofdm_phy_1m.py``.

- Symbols are mapped straight from the sender's [C, H/16, W/16] tensor: per
  layer, the flattened real features in blocks of 96, the first 48 on I and
  the second 48 on Q with alternating signs (DeepStream Eq. 6). The receiver
  inverts this from the image size and the layer table, without any reference.
- Header: magic, image sequence number, packet index and packet count, CRC32
  (80 bits), the IEEE 802.11 convolutional code (K = 7, rate 1/2, generators
  133/171 octal, 6 zero tail bits), 20 known pad bits and a fixed interleaver
  over its 4 OFDM rows (192 BPSK symbols). Soft-decision Viterbi, each carrier
  weighted by its channel power. The re-encoded header gives the SNR estimate.
- A transmission may carry any subset of packets.
- Channel estimate: the LS values of both LTFs, de-ramped by the dominant delay
  and smoothed by the linear smoother (carrier averages of several lengths,
  projections onto short impulse responses) with the lowest Stein unbiased risk
  estimate, from the noise measured between the two LTFs. At 156 kHz carrier
  spacing an indoor channel can be frequency selective, so the fixed 7-carrier
  average of the 1 MS/s receiver would blur it. Pilots: one phase slope per
  packet, a common phase per symbol.
- The TX streams continuously, so its LO leakage is always on the air, at the
  band centre (DC null after CFO correction). Packet detection runs on the
  signal minus its 16-sample moving average: this removes near-DC tones and
  leaves the period-16, DC-free STF unchanged.
- Each payload carrier is rotated by a fixed pseudo-random phase (removed after
  equalization; AWGN statistics are unchanged), which bounds the PAPR of
  spatially correlated features.

numpy + scipy (no torch), so the AGX runs the same code. Both ends work on all
packets of a burst at once (one IFFT, one interpolation call, one Viterbi pass over all
headers); the I/Q samples are bit-identical to the earlier per-packet code. When a C++
compiler is available, :func:`modulate` and :func:`recover` (up to the header CRC check)
run in ``ofdm_native.cpp`` (:func:`native_kernel`; ``native=False`` or ``OFDM_NATIVE=0``
selects numpy). 8 packets (2026-11-08): Jetson modulation 8.7 -> 1.45 ms, RTX-host
recovery 7.7 -> 0.92 ms.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import struct
import subprocess
import warnings
import zlib

import numpy as np
from scipy import signal

RATE = 20_000_000           # HackRF sample rate (signed int8 I/Q)
BASE_RATE = 10_000_000      # OFDM baseband sample rate
UP = RATE // BASE_RATE
OFFSET = 0                  # band centre minus LO: the LO sits on the DC null carrier
NFFT = 64
CP = 16
SYMLEN = NFFT + CP
ACTIVE = np.r_[np.arange(-26, 0), np.arange(1, 27)]
PILOT_K = np.array([-21, -7, 7, 21])
DATA_K = np.array([k for k in ACTIVE if k not in PILOT_K])
ACTIVE_IDX = ACTIVE % NFFT
PILOT_IDX = PILOT_K % NFFT
DATA_IDX = DATA_K % NFFT
ROWS = 32                   # payload OFDM symbols per packet
HEADER_ROWS = 4             # 192 coded header bits
GUARD = 512                 # zero samples between packets (also the noise reference)
SETTLE_SECONDS = 0.02       # leading zeros of a finite (non-streamed) burst for the TX chain to settle
FEATURES = 96               # real features per OFDM row (48 I + 48 Q)
TARGET_RMS = 17.34317263748903
PEAK_LIMIT = 0.75
DEFAULT_PERMUTATION_SEED = 20261106
DEFAULT_SCRAMBLE_SEED = 20261107
# STF detection, two stages. (1) Normalized lag-16 autocorrelation over STF_WINDOW products
# (the 160-sample STF gives 144), ~ (snr / (1 + snr))^2 for the SNR in the 10 MHz detection
# band; in noise it is ~exponential with mean 1 / STF_WINDOW, so 0.12 over 128 products is a
# ~2e-7 per-sample false-peak rate. (2) Each peak's normalized cross-correlation with the STF,
# ~ snr / (1 + snr), must exceed STF_SCORE_THRESHOLD (in noise ~0.03). Both only bound the
# work: a false candidate is rejected by the header CRC. Until 2026-11-10 the values were
# 64 / 0.3 / 0.4, which sat at the mean of both statistics at 1-3 dB header SNR: on the saved
# 10 MHz captures below 3 dB, 99 % of the lost packets were dropped here (97 % by stage 2)
# although their headers decode (analyze_loss_causes.py).
STF_WINDOW = 128
STF_THRESHOLD = 0.12
STF_SCORE_THRESHOLD = 0.2
SIGNS = np.where(np.arange(48) % 2 == 0, 1.0, -1.0)  # Eq. 6 with one-based indices
PILOTS = (2 * np.random.default_rng(245064).integers(0, 2, size=HEADER_ROWS + ROWS) - 1)[:, None] \
    * np.array([1, 1, 1, -1])
LTF_GRID = np.zeros(NFFT, complex)
LTF_GRID[ACTIVE_IDX] = 2 * np.random.default_rng(64052).integers(0, 2, len(ACTIVE)) - 1
LTF = np.fft.ifft(LTF_GRID, norm="ortho")
_STF_K = np.r_[np.arange(-24, 0, 4), np.arange(4, 25, 4)]
STF_GRID = np.zeros(NFFT, complex)
STF_GRID[_STF_K % NFFT] = np.sqrt(52 / 12) * (
    2 * np.random.default_rng(64012).integers(0, 2, len(_STF_K)) - 1) * (1 + 1j) / np.sqrt(2)
STF = np.tile(np.fft.ifft(STF_GRID, norm="ortho")[:16], 10)
TRAIN = np.r_[STF, LTF[-32:], LTF, LTF]
PACKET_LEN = len(TRAIN) + (HEADER_ROWS + ROWS) * SYMLEN
HEADER = struct.Struct(">2sHBB")         # magic, seq, packet, total
MAGIC = b"OJ"


# ----------------------------------------------------------------- mapping

def rows_per_layer(layer_channels, height: int, width: int) -> list[int]:
    plane = (height // 16) * (width // 16)
    counts = []
    for c0, c1 in layer_channels:
        if (c1 - c0) * plane % FEATURES:
            raise ValueError(f"Layer {c0}:{c1} at {height}x{width} is not a multiple of {FEATURES} features.")
        counts.append((c1 - c0) * plane // FEATURES)
    return counts


def symbols_to_rows(symbols: np.ndarray, layer_channels) -> np.ndarray:
    """Sender tensor [C, h, w] (per-layer unit complex power) -> OFDM data rows [N, 48]."""
    rows = []
    for c0, c1 in layer_channels:
        blocks = np.asarray(symbols[c0:c1], np.float64).reshape(-1, FEATURES)
        rows.append(blocks[:, :48] + 1j * blocks[:, 48:] * SIGNS)
    return np.concatenate(rows)


def rows_to_symbols(rows: np.ndarray, layer_channels, height: int, width: int) -> np.ndarray:
    """Exact inverse of :func:`symbols_to_rows`; missing rows stay NaN."""
    h, w = height // 16, width // 16
    out = np.empty((layer_channels[-1][1], h, w), np.float32)
    cursor = 0
    for (c0, c1), count in zip(layer_channels, rows_per_layer(layer_channels, height, width)):
        block = rows[cursor:cursor + count]
        out[c0:c1] = np.concatenate([block.real, block.imag * SIGNS], axis=1).reshape(c1 - c0, h, w)
        cursor += count
    if not 0 <= len(rows) - cursor < ROWS:  # air_rows pads the last packet with filler rows
        raise ValueError("Row count does not match the layer table.")
    return out


def payload_to_rows(values: np.ndarray, kind: str, filler_seed: int = 0) -> np.ndarray:
    """Generic payload -> OFDM data rows filling whole packets.

    ``kind="real"``: real JSCC features, 96 per row as in :func:`symbols_to_rows`.
    ``kind="complex"``: complex symbols (e.g. QAM), one per data subcarrier.
    The tail of the last packet carries deterministic unit-power QPSK filler,
    so the transmit power per symbol is the same as for a full payload.
    """
    values = np.asarray(values).reshape(-1)
    if kind == "real":
        if len(values) % FEATURES:
            raise ValueError(f"Real payload length must be a multiple of {FEATURES}.")
        blocks = values.astype(np.float64).reshape(-1, FEATURES)
        rows = blocks[:, :48] + 1j * blocks[:, 48:] * SIGNS
    elif kind == "complex":
        flat = values.astype(np.complex128)
        rows = np.r_[flat, np.zeros((-len(flat)) % 48, complex)].reshape(-1, 48)
        tail = (-len(flat)) % 48
        if tail:
            rows.reshape(-1)[len(flat):] = _qpsk(tail, filler_seed + 1)
    else:
        raise ValueError(f"Unknown payload kind {kind!r}")
    extra = (-len(rows)) % ROWS if len(rows) else ROWS  # an empty payload still sends one packet
    if extra:
        rows = np.r_[rows.reshape(-1, 48), _qpsk(extra * 48, filler_seed).reshape(extra, 48)]
    return rows


def rows_to_payload(rows: np.ndarray, kind: str, n_values: int) -> np.ndarray:
    """Inverse of :func:`payload_to_rows` (rows of missing packets give NaN)."""
    if kind == "real":
        return np.concatenate([rows.real, rows.imag * SIGNS], axis=1).reshape(-1)[:n_values].astype(np.float32)
    if kind == "complex":
        return rows.reshape(-1)[:n_values].astype(np.complex64)
    raise ValueError(f"Unknown payload kind {kind!r}")


FILLER_SEED = 20261109  # QPSK filler completing a partly filled last packet (air_rows)


def _qpsk(count: int, seed: int) -> np.ndarray:
    bits = np.random.default_rng(seed).integers(0, 2, (count, 2))
    return ((2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2)


def probe_rows(seed: int) -> np.ndarray:
    """Two packets of unit-power complex Gaussian rows (same statistics as JSCC symbols).

    Deterministic in ``seed`` (the transmission's sequence number), so the receiver
    knows the probe payload and estimates the SNR from all 3072 of its symbols.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2 * ROWS, 48)) + 1j * rng.standard_normal((2 * ROWS, 48))) / np.sqrt(2)


def permutation(seed: int | None) -> np.ndarray:
    """Fixed permutation of the 48 data carriers inside every payload row (None = identity)."""
    return np.arange(48) if seed is None else np.random.default_rng(seed).permutation(48)


def permute_rows(rows: np.ndarray, perm: np.ndarray, inverse: bool = False) -> np.ndarray:
    perm = np.asarray(perm)
    if perm.shape != (48,) or not np.array_equal(np.sort(perm), np.arange(48)):
        raise ValueError("Expected a permutation of the 48 data carriers.")
    return rows[:, np.argsort(perm) if inverse else perm]


def scrambling(n_rows: int, seed: int | None) -> np.ndarray:
    """Unit-modulus phase per (frame row, carrier); ones when ``seed`` is None."""
    if seed is None:
        return np.ones((n_rows, 48))
    return np.exp(2j * np.pi * np.random.default_rng(seed).random((n_rows, 48)))


def air_rows(symbols: np.ndarray, layer_channels, permutation_seed: int | None = DEFAULT_PERMUTATION_SEED,
             scramble_seed: int | None = DEFAULT_SCRAMBLE_SEED) -> np.ndarray:
    """Sender tensor -> on-air payload rows: feature mapping, carrier permutation, phase scrambling.

    Rows that do not fill the last packet (e.g. CR 768 at 768x1280: 80 rows, packets of 32) are
    completed with unit-power QPSK filler rows; :func:`rows_to_symbols` ignores them."""
    rows = symbols_to_rows(symbols, layer_channels)
    pad = -len(rows) % ROWS
    if pad:
        rows = np.concatenate([rows, _qpsk(pad * 48, seed=FILLER_SEED).reshape(pad, 48)])
    return to_air(rows, permutation_seed, scramble_seed)


def to_air(rows: np.ndarray, permutation_seed: int | None = DEFAULT_PERMUTATION_SEED,
           scramble_seed: int | None = DEFAULT_SCRAMBLE_SEED) -> np.ndarray:
    """Data rows -> on-air rows (carrier permutation, then phase scrambling); see :func:`assemble`."""
    rows = permute_rows(rows, permutation(permutation_seed))
    return rows * scrambling(len(rows), scramble_seed)


# ------------------------------------------------------------- header code

CONV_K = 7
CONV_G = (0o133, 0o171)                     # IEEE 802.11 generators
TAIL_BITS = CONV_K - 1
HEADER_BITS = 8 * (HEADER.size + 4)         # 80: fields + CRC32
CODED_BITS = 2 * (HEADER_BITS + TAIL_BITS)  # 172
HEADER_SYMBOLS = HEADER_ROWS * 48           # 192
HEADER_PAD = np.random.default_rng(20261108).integers(0, 2, HEADER_SYMBOLS - CODED_BITS).astype(np.uint8)
HEADER_INTERLEAVER = np.random.default_rng(20261109).permutation(HEADER_SYMBOLS)  # air[i] = coded[perm[i]]
_STATES = 1 << (CONV_K - 1)


def _conv_tables():
    """Register = (input << 6) | state; outputs = parity(register & g); next state = register >> 1."""
    nxt = np.zeros((_STATES, 2), np.int64)
    out = np.zeros((_STATES, 2, 2), np.int64)
    for state in range(_STATES):
        for bit in (0, 1):
            register = (bit << (CONV_K - 1)) | state
            nxt[state, bit] = register >> 1
            out[state, bit] = [bin(register & g).count("1") & 1 for g in CONV_G]
    return nxt, out


_CONV_NEXT, _CONV_OUT = _conv_tables()
_CONV_SIGN = 2.0 * _CONV_OUT - 1.0          # [state, input, output] in BPSK (+1 = bit 1)


def _conv_encode_loop(bits: np.ndarray) -> np.ndarray:
    state, coded = 0, []
    for bit in np.r_[np.asarray(bits, np.int64), np.zeros(TAIL_BITS, np.int64)]:
        coded.extend(_CONV_OUT[state, bit])
        state = _CONV_NEXT[state, bit]
    return np.asarray(coded, np.uint8)


# The code is linear and starts in state 0: row i is the codeword of the i-th unit vector.
_CONV_GENERATOR = np.stack([_conv_encode_loop(np.eye(HEADER_BITS, dtype=np.int64)[i])
                            for i in range(HEADER_BITS)]).astype(np.int32)


def conv_encode(bits: np.ndarray) -> np.ndarray:
    """Rate-1/2 K=7 code of ``bits`` followed by 6 zero tail bits (terminated in state 0).

    ``bits``: [..., n]; header-length words (the only ones on the air) use the generator
    matrix, so a batch of headers is one matrix product.
    """
    bits = np.asarray(bits, np.int64)
    if bits.shape[-1] == HEADER_BITS:
        return ((bits.astype(np.int32) @ _CONV_GENERATOR) & 1).astype(np.uint8)
    if bits.ndim != 1:
        return np.stack([conv_encode(b) for b in bits.reshape(-1, bits.shape[-1])]).reshape(
            *bits.shape[:-1], -1)
    return _conv_encode_loop(bits)


def viterbi(soft: np.ndarray) -> np.ndarray:
    """Soft-decision Viterbi for :func:`conv_encode` (terminated trellis).

    ``soft``: [..., 2 * steps], 2 values per encoded bit, positive = 1 (BPSK, weighted by
    reliability). Returns [..., steps], all decoded bits including the tail. A next state
    ns = (input << 5) | j has the predecessors 2j and 2j + 1, so one step is a few
    [batch, 2, 32] array operations: a batch of headers costs about as much as one.
    """
    soft = np.asarray(soft, np.float64)
    batch = soft.shape[:-1]
    soft = soft.reshape(-1, soft.shape[-1] // 2, 2)                                   # [b, t, k]
    count, steps = soft.shape[:2]
    branch = (soft[:, :, None, None, 0] * _CONV_SIGN[:, :, 0]
              + soft[:, :, None, None, 1] * _CONV_SIGN[:, :, 1]).reshape(count, steps, 32, 2, 2)  # [n, t, j, x, input]
    branch = np.ascontiguousarray(branch.transpose(1, 0, 4, 2, 3))  # [t, n, input, j, x]: predecessor 2j + x
    metric = np.full((count, _STATES), -np.inf)
    metric[:, 0] = 0.0
    decisions = np.empty((steps, count, _STATES), bool)
    for t in range(steps):
        pairs = metric.reshape(count, 1, 32, 2)
        even = pairs[..., 0] + branch[t, ..., 0]                    # [n, input, j]
        odd = pairs[..., 1] + branch[t, ..., 1]
        np.greater(odd, even, out=decisions[t].reshape(count, 2, 32))
        metric = np.maximum(even, odd).reshape(count, _STATES)
    bits = np.empty((count, steps), np.uint8)
    state, rows = np.zeros(count, np.int64), np.arange(count)
    for t in range(steps - 1, -1, -1):
        bits[:, t] = state >> 5
        state = ((state & 31) << 1) | decisions[t, rows, state]
    return bits.reshape(*batch, steps)


def header_bits(seq: int, packet: int, total: int) -> np.ndarray:
    body = HEADER.pack(MAGIC, seq & 0xFFFF, packet, total)
    word = body + struct.pack(">I", zlib.crc32(body))
    return np.unpackbits(np.frombuffer(word, dtype=np.uint8))


def header_symbols(seq, packet, total: int) -> np.ndarray:
    """BPSK header rows [HEADER_ROWS, 48] of one packet, or [n, HEADER_ROWS, 48] when
    ``packet`` (and optionally ``seq``) are sequences."""
    if np.ndim(packet) == 0:
        return header_symbols([seq], [packet], total)[0]
    seqs = np.broadcast_to(seq, np.shape(packet))
    bits = np.stack([header_bits(int(s), int(p), total) for s, p in zip(seqs, packet)])
    coded = np.concatenate([conv_encode(bits), np.broadcast_to(HEADER_PAD, (len(bits), len(HEADER_PAD)))], axis=1)
    return (2.0 * coded[:, HEADER_INTERLEAVER] - 1.0).reshape(len(bits), HEADER_ROWS, 48)


def _check_header(bits: np.ndarray, total: int, seq: int | None):
    word = np.packbits(bits[:HEADER_BITS]).tobytes()
    if zlib.crc32(word[:HEADER.size]) != struct.unpack(">I", word[HEADER.size:])[0]:
        return None
    magic, got_seq, index, count = HEADER.unpack(word[:HEADER.size])
    if magic != MAGIC or (seq is not None and got_seq != seq & 0xFFFF) or count != total or index >= total:
        return None
    return index, got_seq


def _header_soft(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Header rows [..., >= HEADER_ROWS, 48] -> de-interleaved soft code bits [..., CODED_BITS]."""
    soft = values[..., :HEADER_ROWS, :].real
    if weights is not None:
        soft = soft * weights[..., None, :]
    coded = np.empty((*soft.shape[:-2], HEADER_SYMBOLS))
    coded[..., HEADER_INTERLEAVER] = soft.reshape(*soft.shape[:-2], -1)
    return coded[..., :CODED_BITS]


def parse_header(values: np.ndarray, total: int, seq: int | None, weights: np.ndarray | None = None):
    """Viterbi-decode the header rows, check CRC, magic, seq and total. Returns (packet, seq) or None.

    ``weights``: per data carrier reliability (channel power for ZF-equalized values).
    """
    return _check_header(viterbi(_header_soft(values, weights)), total, seq)


PACKET_SPACING = GUARD + PACKET_LEN  # baseband samples between packet starts within one burst (see _intervals)
INFER_TOLERANCE = 4                 # samples: a header-failed packet must sit on the burst's packet grid
INFER_MIN_BIT_AGREEMENT = 0.85      # decoded header bits vs the header expected at that slot


def _infer_packet_ids(parsed, usable, starts, quality, bits, total: int, seq: int | None) -> dict:
    """Packets whose header failed the CRC, identified by position (``recover(infer_ids=True)``).

    The packets of a burst start every ``PACKET_SPACING`` baseband samples, so the CRC-valid
    packets give the burst's grid origin; a usable candidate whose header failed is taken as the
    packet of its grid slot when it lies within ``INFER_TOLERANCE`` samples of the slot, the slot
    is in range and still empty, and its Viterbi-decoded header bits agree with the header that
    slot should carry (seq of the valid packets, slot id, ``total``) in at least
    ``INFER_MIN_BIT_AGREEMENT`` of the 80 bits. A CRC failure from a few bit errors passes; a
    packet of another transmission (other seq, so ~half of the 32 CRC bits differ) or a noise
    detection does not. Returns {candidate index: (packet id, seq)}.
    """
    accepted = [i for i, p in enumerate(parsed) if p is not None]
    if not accepted:
        return {}
    frame_seq = int(parsed[accepted[0]][1]) if seq is None else int(seq) & 0xFFFF
    origin = float(np.median([starts[i] - parsed[i][0] * PACKET_SPACING for i in accepted]))
    taken = {parsed[i][0] for i in accepted}
    out = {}
    for i in sorted((i for i, p in enumerate(parsed) if p is None and usable[i]), key=lambda i: -quality[i]):
        slot = int(round((starts[i] - origin) / PACKET_SPACING))
        if not 0 <= slot < total or slot in taken or abs(starts[i] - origin - slot * PACKET_SPACING) > INFER_TOLERANCE:
            continue
        agreement = float(np.mean(np.asarray(bits[i][:HEADER_BITS]) == header_bits(frame_seq, slot, total)))
        if agreement < INFER_MIN_BIT_AGREEMENT:
            continue
        out[i] = (slot, frame_seq)
        taken.add(slot)
    return out


# ------------------------------------------------------------- transmitter

def _ofdm_rows(data: np.ndarray) -> np.ndarray:
    """Data rows [..., symbols, 48] -> time samples with cyclic prefixes [..., symbols * SYMLEN]."""
    grid = np.zeros((*data.shape[:-1], NFFT), complex)
    grid[..., DATA_IDX] = data
    grid[..., PILOT_IDX] = PILOTS[:data.shape[-2]]
    time = np.fft.ifft(grid, axis=-1, norm="ortho")
    return np.concatenate([time[..., -CP:], time], axis=-1).reshape(*data.shape[:-2], -1)


BLOCK_LEN = PACKET_LEN  # training + header + payload samples of one packet


def baseband_waveform(rows: np.ndarray, seq: int, packet_ids=None, lead: int = 0):
    """Baseband waveform of the selected packets; returns (samples, active intervals).

    ``lead`` zero samples come first (a finite burst needs ``SETTLE_SECONDS``; a
    continuous TX stream needs none), then for every packet a guard and the packet,
    then a closing guard. All packets are built at once (one IFFT for the burst).
    """
    blocks = _packet_blocks(rows, seq, packet_ids)
    slots = np.zeros((len(blocks), GUARD + BLOCK_LEN), complex)  # guard, then packet
    slots[:, GUARD:] = blocks
    low = np.concatenate([np.zeros(lead), slots.reshape(-1), np.zeros(GUARD)])
    return low, _intervals(len(blocks), lead)


def _packet_data(rows: np.ndarray, seq: int, packet_ids=None) -> np.ndarray:
    """Header + payload rows [packets, HEADER_ROWS + ROWS, 48] of the selected packets."""
    if len(rows) % ROWS:
        raise ValueError(f"{len(rows)} rows do not fill whole packets of {ROWS}.")
    total = len(rows) // ROWS
    packet_ids = list(range(total)) if packet_ids is None else [int(p) for p in packet_ids]
    if not packet_ids or len(set(packet_ids)) != len(packet_ids) or not all(0 <= p < total for p in packet_ids):
        raise ValueError(f"Invalid packet ids {packet_ids} for {total} packets.")
    ids = np.asarray(packet_ids)
    payload = np.asarray(rows).reshape(total, ROWS, 48)[ids]
    return np.concatenate([header_symbols(seq, ids, total), payload], axis=1)


def _packet_blocks(rows: np.ndarray, seq: int, packet_ids=None) -> np.ndarray:
    """Training + header + payload samples [packets, BLOCK_LEN] of the selected packets."""
    data = _packet_data(rows, seq, packet_ids)
    blocks = np.empty((len(data), BLOCK_LEN), complex)
    blocks[:, :len(TRAIN)] = TRAIN
    blocks[:, len(TRAIN):] = _ofdm_rows(data)
    return blocks


def _intervals(packets: int, lead: int) -> list[tuple[int, int]]:
    """Active spans (baseband samples) of a burst: lead, then (guard, packet) per packet, then a guard."""
    starts = lead + GUARD + np.arange(packets) * (GUARD + BLOCK_LEN)
    return [(int(a), int(a) + BLOCK_LEN) for a in starts]


def pack_iq(wide: np.ndarray, intervals, target_rms: float = TARGET_RMS, peak_limit: float = PEAK_LIMIT):
    """One global gain: active RMS -> target counts, capped so the complex peak stays <= peak_limit.

    ``intervals`` are the active packet spans in baseband samples. The JSCC
    features themselves are never clipped.
    """
    if not 0.75 / 128 < peak_limit <= 1 or not np.all(np.isfinite(wide)) or target_rms <= 0:
        raise ValueError("Finite samples and a normalized peak limit in (0, 1] are required.")
    re, im = wide.real, wide.imag
    power = re * re + im * im
    rms = float(np.sqrt(np.mean(np.concatenate([power[a * UP:b * UP] for a, b in intervals]))))
    peak = float(np.sqrt(power.max()))
    # <= sqrt(0.5) count is the largest complex rounding displacement
    gain = min(target_rms / rms, (128 * peak_limit - 0.75) / peak)
    i, q = np.rint(re * gain), np.rint(im * gain)
    limit = float(np.sqrt(np.max(i * i + q * q)))
    if limit > 127 or limit / 128 > peak_limit + 1e-12:
        raise AssertionError("Quantized I/Q exceeds the peak limit.")
    packed = np.empty((len(wide), 2), np.int8)
    packed[:, 0], packed[:, 1] = i, q
    return packed, {"digital_gain": float(gain), "active_rms_counts": float(rms * gain),
                    "normalized_peak": limit / 128, "rms_limited_by_peak": bool(gain < target_rms / rms),
                    "papr_active_db": float(10 * np.log10(peak ** 2 / rms ** 2))}


def interpolate(low: np.ndarray, intervals) -> np.ndarray:
    """``resample_poly(low, UP, 1)`` evaluated only around the packets.

    The filter spans +-10 baseband samples, and the waveform is exactly zero
    between packets (guards), so each packet with 16 zero samples of context
    gives the same values; elsewhere the output is 0. Spans of equal length are
    filtered in one call (one filter design, one pass), with the real and imaginary
    parts as separate real rows: the filter is real, so this gives bit-identical
    values 3x faster than scipy's complex path.
    """
    wide = np.zeros(len(low) * UP, complex)
    pad = 16
    spans = [(max(0, a - pad), min(len(low), b + pad)) for a, b in intervals]
    width = max(hi - lo for lo, hi in spans)
    full = [(lo, hi) for lo, hi in spans if hi - lo == width]
    if full:
        segments = low[np.array([lo for lo, _ in full])[:, None] + np.arange(width)]
        out = signal.resample_poly(np.concatenate([segments.real, segments.imag]), UP, 1, axis=1)
        out = out[:len(full)] + 1j * out[len(full):]
        for (lo, hi), values in zip(full, out):
            wide[lo * UP:hi * UP] = values
    for lo, hi in spans:  # spans clipped at the ends of the burst
        if hi - lo != width:
            wide[lo * UP:hi * UP] = signal.resample_poly(low[lo:hi], UP, 1)
    return wide


def _interpolation_taps():
    """Taps of ``resample_poly(x, UP, 1)``: y[UP n + r] = sum_i taps[i, r] x[n + i - left].

    Read off its impulse response, so they follow the installed scipy exactly."""
    middle = 64
    phases = signal.resample_poly(np.eye(2 * middle + 1)[middle], UP, 1).reshape(-1, UP)  # [n, r]
    support = np.flatnonzero(np.any(phases != 0, axis=1))
    lo, hi = int(support[0]), int(support[-1])
    return np.ascontiguousarray(phases[hi:lo - 1 if lo else None:-1]), hi - middle


INTERP_TAPS, INTERP_LEFT = _interpolation_taps()
# For the C++ kernel: the even output phase of this half-band filter is the centre tap alone;
# its other taps (~1e-17) are rounding residues of the sinc zeros and are dropped (zero = skipped).
_NATIVE_TAPS = np.where(np.abs(INTERP_TAPS) < 1e-12 * np.abs(INTERP_TAPS).max(), 0.0, INTERP_TAPS)
_NATIVE_SOURCE = Path(__file__).with_name("ofdm_native.cpp")
_native_lib = None  # ctypes library once loaded, False when unavailable


def native_kernel():
    """The C++ transmitter kernel (``ofdm_native.cpp``), compiled on first use into
    ``$OFDM_NATIVE_DIR`` (default ``~/.cache/osd_ofdm``) under the hash of its source.
    None when ``OFDM_NATIVE=0`` or no compiler works: :func:`modulate` then uses numpy."""
    global _native_lib
    if _native_lib is None:
        _native_lib = False
        if os.environ.get("OFDM_NATIVE", "1") != "0" and _NATIVE_SOURCE.exists():
            try:
                digest = hashlib.sha1(_NATIVE_SOURCE.read_bytes()).hexdigest()[:12]
                cache = Path(os.environ.get("OFDM_NATIVE_DIR", Path.home() / ".cache" / "osd_ofdm"))
                path = cache / f"ofdm_native_{digest}_{platform.system()}_{platform.machine()}.so"
                if not path.exists():
                    cache.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                    subprocess.run([os.environ.get("CXX", "c++"), "-O3", "-ffp-contract=off", "-std=c++17",
                                    "-shared", "-fPIC", "-o", str(tmp), str(_NATIVE_SOURCE)],
                                   check=True, capture_output=True, text=True)
                    os.replace(tmp, path)
                    for stale in cache.glob(f"ofdm_native_*_{platform.system()}_{platform.machine()}.so"):
                        if stale != path:  # builds of earlier versions of the source
                            stale.unlink(missing_ok=True)
                lib = ctypes.CDLL(str(path))
                f64, i64 = np.ctypeslib.ndpointer(np.float64, flags="C"), np.ctypeslib.ndpointer(np.int64, flags="C")
                u8, i32 = np.ctypeslib.ndpointer(np.uint8, flags="C"), np.ctypeslib.ndpointer(np.int32, flags="C")
                cint, ci64, cdbl = ctypes.c_int, ctypes.c_int64, ctypes.c_double
                lib.ofdm_modulate.restype = cint
                lib.ofdm_modulate.argtypes = [
                    f64, cint, cint, f64, cint, f64, i64, i64, cint, f64, cint, cint, cint, cdbl, cdbl,
                    np.ctypeslib.ndpointer(np.int8, flags="C"), ci64, ci64, f64]
                lib.ofdm_rx_setup.restype = cint
                lib.ofdm_rx_setup.argtypes = [f64, cint, f64, cint, f64, i64, i64, i64, f64, cint, cint, i64, i64,
                                              cint, f64, f64, f64, i64, cint, cint, f64, f64, i64]
                lib.ofdm_rx_recover.restype = ci64
                lib.ofdm_rx_recover.argtypes = [f64, ci64, cint, cint, cint, ci64, f64, i64, i64, f64, f64, f64,
                                                f64, f64, i32, u8, f64, u8, f64]
                _native_rx_setup(lib)
                _native_lib = lib
            except (OSError, subprocess.CalledProcessError) as error:
                detail = getattr(error, "stderr", "") or error
                warnings.warn(f"C++ OFDM kernel unavailable, using numpy: {detail}")
    return _native_lib or None


def _native_rx_setup(lib) -> None:
    """Hand the receiver tables and constants of this module to the C++ kernel."""
    projections = [f for f in _SMOOTHER_FACTORS if f[0] == "projection"]
    as_f64 = lambda a: np.ascontiguousarray(a, complex).reshape(-1).view(np.float64)  # noqa: E731
    params = [f[1] if f[0] == "box" else f[1].shape[1] for f in _SMOOTHER_FACTORS]
    ints = np.array([CP, ADVANCE, len(TRAIN), PACKET_LEN, 448, 64, 16, STF_WINDOW, 120, 200, 400, 100, 64, 20, 32], np.int64)
    code = lib.ofdm_rx_setup(
        as_f64(STF), len(STF), as_f64(TRAIN[len(STF):]), len(TRAIN) - len(STF), as_f64(LTF_GRID[ACTIVE_IDX]),
        ACTIVE.astype(np.int64), DATA_K.astype(np.int64), PILOT_K.astype(np.int64),
        np.ascontiguousarray(PILOTS, np.float64), HEADER_ROWS + ROWS, HEADER_ROWS,
        np.array([0 if f[0] == "box" else 1 for f in _SMOOTHER_FACTORS], np.int64), np.array(params, np.int64),
        len(_SMOOTHER_FACTORS), as_f64(np.concatenate([f[1].reshape(-1) for f in projections])),
        as_f64(np.concatenate([f[2].reshape(-1) for f in projections])),
        np.ascontiguousarray(_SMOOTHER_TRACE, np.float64), HEADER_INTERLEAVER.astype(np.int64),
        len(HEADER_INTERLEAVER), CODED_BITS, np.ascontiguousarray(_CONV_SIGN, np.float64),
        np.array([BASE_RATE, STF_THRESHOLD, STF_SCORE_THRESHOLD]), ints)
    if code:
        raise OSError(f"ofdm_rx_setup failed ({code})")


_TRAIN_IQ = np.ascontiguousarray(TRAIN.astype(complex)).view(np.float64)
_PILOTS_F64 = np.ascontiguousarray(PILOTS, np.float64)
_DATA_IDX64, _PILOT_IDX64 = DATA_IDX.astype(np.int64), PILOT_IDX.astype(np.int64)


def modulate(rows: np.ndarray, seq: int, packet_ids=None, *, lead: int = 0,
             target_rms: float = TARGET_RMS, peak_limit: float = PEAK_LIMIT, native: bool | None = None):
    """OFDM rows -> HackRF int8 I/Q [N, 2] at ``RATE`` (band centred on the LO) and its metadata.

    ``info["packet_starts"]``: first sample (at ``RATE``) of every packet's STF in the
    burst, in transmission order (lets the receiver predict where a streamed burst lands).
    The numpy path equals ``pack_iq(interpolate(*baseband_waveform(...)))`` bit for bit, but
    only the packets and their filter context are computed: the zero guards stay zero.
    ``native`` (default: when it compiles) runs the same steps in C++ (``ofdm_native.cpp``);
    its own FFT and sums move samples by ~1e-15, so a sample can differ by one count where
    the scaled value falls on a rounding boundary.
    """
    if not 0.75 / 128 < peak_limit <= 1 or target_rms <= 0:
        raise ValueError("A normalized peak limit in (0, 1] and a positive target RMS are required.")
    pad, pitch = 16, GUARD + BLOCK_LEN
    lib = native_kernel() if native is None or native else None
    if native and lib is None:
        raise RuntimeError("The C++ OFDM kernel is not available.")
    if lib is not None:
        data = np.ascontiguousarray(_packet_data(rows, seq, packet_ids), complex)
        count = len(data)
        intervals = _intervals(count, lead)
        packed = np.zeros(((lead + count * pitch + GUARD) * UP, 2), np.int8)
        stats = np.empty(4)
        code = lib.ofdm_modulate(data.reshape(-1).view(np.float64), count, data.shape[1], _TRAIN_IQ, len(TRAIN),
                                 _PILOTS_F64, _DATA_IDX64, _PILOT_IDX64, CP, _NATIVE_TAPS, len(_NATIVE_TAPS),
                                 INTERP_LEFT, pad, float(target_rms), float(peak_limit), packed,
                                 (intervals[0][0] - pad) * UP, pitch * UP, stats)
        if code == 1:
            raise ValueError("Finite samples are required.")
        if code == 2:
            raise AssertionError("Quantized I/Q exceeds the peak limit.")
        gain, rms, peak, limit = (float(v) for v in stats)
        return packed, _modulate_info(gain, rms, peak, limit, target_rms, intervals, len(packed))
    blocks = _packet_blocks(rows, seq, packet_ids)
    count = len(blocks)
    intervals = _intervals(count, lead)
    # each packet with 16 zero samples either side (interpolate()), real and imaginary rows
    segments = np.zeros((2 * count, BLOCK_LEN + 2 * pad))
    segments[:count, pad:-pad], segments[count:, pad:-pad] = blocks.real, blocks.imag
    up = signal.resample_poly(segments, UP, 1, axis=1)
    if not np.all(np.isfinite(up)):
        raise ValueError("Finite samples are required.")
    re, im = up[:count], up[count:]
    power = re * re + im * im
    rms = float(np.sqrt(np.mean(np.ascontiguousarray(power[:, pad * UP:(pad + BLOCK_LEN) * UP]).reshape(-1))))
    peak = float(np.sqrt(power.max()))
    gain = min(target_rms / rms, (128 * peak_limit - 0.75) / peak)  # as pack_iq
    i, q = np.rint(re * gain), np.rint(im * gain)
    limit = float(np.sqrt(np.max(i * i + q * q)))
    if limit > 127 or limit / 128 > peak_limit + 1e-12:
        raise AssertionError("Quantized I/Q exceeds the peak limit.")
    packed = np.zeros(((lead + count * pitch + GUARD) * UP, 2), np.int8)
    first = (intervals[0][0] - pad) * UP
    region = packed[first:first + count * pitch * UP].reshape(count, pitch * UP, 2)
    region[:, :up.shape[1], 0], region[:, :up.shape[1], 1] = i, q
    return packed, _modulate_info(gain, rms, peak, limit, target_rms, intervals, len(packed))


def _modulate_info(gain, rms, peak, limit, target_rms, intervals, samples) -> dict:
    return {"digital_gain": float(gain), "active_rms_counts": float(rms * gain),
            "normalized_peak": limit / 128, "rms_limited_by_peak": bool(gain < target_rms / rms),
            "papr_active_db": float(10 * np.log10(peak ** 2 / rms ** 2)),
            "samples": samples, "seconds": samples / RATE,
            "active_seconds": sum(b - a for a, b in intervals) / BASE_RATE,
            "packets": len(intervals), "packet_starts": [int(a * UP) for a, _ in intervals]}


# ---------------------------------------------------------------- receiver

def to_baseband(iq: np.ndarray) -> np.ndarray:
    """HackRF int8 I/Q (interleaved or [N, 2]) -> complex baseband at ``BASE_RATE``."""
    pairs = np.asarray(iq, np.int8).reshape(-1, 2)
    z = pairs[:, 0].astype(np.float32) + 1j * pairs[:, 1].astype(np.float32)
    return signal.resample_poly(z, 1, UP)


def _moving_sum(values: np.ndarray, length: int) -> np.ndarray:
    cumulative = np.r_[0, np.cumsum(values)]
    return cumulative[length:] - cumulative[:-length]


def detection_signal(z: np.ndarray) -> np.ndarray:
    """``z`` minus its 16-sample moving average.

    Removes near-DC tones (the TX LO leakage that the continuous TX stream radiates
    at the band centre would otherwise make the lag-16 autocorrelation fire on idle
    stretches) and leaves the STF, which has period 16 and no DC, unchanged.
    """
    out = z.copy()
    if len(z) >= 16:
        out[8:len(z) - 7] -= _moving_sum(z, 16) / 16
    return out


ADVANCE = 4  # FFT windows start 4 samples inside the CP (LTF and data alike)


def _channel_ramp(h: np.ndarray) -> np.ndarray:
    """Linear phase across the active carriers from the dominant (possibly fractional) delay.

    ``h``: [..., active carriers]; one ramp per leading index."""
    step = np.diff(ACTIVE) == 1
    slope = np.angle(np.sum(np.conj(h[..., :-1][..., step]) * h[..., 1:][..., step], axis=-1))
    return np.exp(1j * slope[..., None] * ACTIVE)


def _smoothers():
    """Linear channel smoothers, applied to the de-ramped LS values: moving averages over
    L adjacent active carriers (edges average fewer) and projections onto impulse responses
    with taps -1..b around the dominant delay (b up to CP - 1 - ADVANCE: every path that
    causes no inter-symbol interference)."""
    n = len(ACTIVE)
    mats, names, factors = [], [], []
    for length in (1, 3, 5, 7, 9, 13, 19, 27, n):
        m = np.zeros((n, n))
        for k in range(n):
            lo, hi = max(0, k - length // 2), min(n, k + length // 2 + 1)
            m[k, lo:hi] = 1 / (hi - lo)
        mats.append(m.astype(complex))
        names.append("ls" if length == 1 else f"avg{length}")
        factors.append(("box", length))
    for last in (1, 2, 3, 4, 6, 8, 10, CP - 1 - ADVANCE):
        basis = np.exp(-2j * np.pi * np.outer(ACTIVE, np.arange(-1, last + 1)) / NFFT)
        pinv = np.linalg.pinv(basis)
        mats.append(basis @ pinv)
        names.append(f"taps-1..{last}")
        factors.append(("projection", basis, pinv))  # the C++ receiver applies basis @ (pinv @ g)
    mats = np.stack(mats)
    return mats, np.real(np.trace(mats, axis1=1, axis2=2)), names, factors


SMOOTHERS, _SMOOTHER_TRACE, SMOOTHER_NAMES, _SMOOTHER_FACTORS = _smoothers()
_SMOOTHER_STACK = SMOOTHERS.reshape(-1, len(ACTIVE))  # all smoothers as one matrix (one product per batch)


def _smooth_channel(h: np.ndarray, taps: int) -> np.ndarray:
    """Moving average of the LS channel over ``taps`` adjacent active carriers (1 MS/s receiver).

    The FFT window starts inside the CP, so H has a linear phase across carriers;
    it is removed before averaging and restored after. Edges average fewer taps.
    """
    if taps <= 1:
        return h
    ramp = _channel_ramp(h)
    kernel = np.ones(taps)
    flat = np.convolve(h / ramp, kernel, mode="same") / np.convolve(np.ones(len(h)), kernel, mode="same")
    return flat * ramp


def _channel_sure(h: np.ndarray, noise_var: float) -> tuple[np.ndarray, str]:
    """Channel estimate: the smoother with the lowest Stein unbiased risk estimate.

    For a linear smoother S of LS values with noise variance s2 per carrier,
    E|S h - H|^2 = E|S h - h|^2 + 2 s2 tr(S) - n s2 (SURE). A flat channel at low
    SNR picks a long average, a frequency-selective one a short average or a
    delay-window projection; ``noise_var`` comes from the difference of the two LTFs.
    """
    channel, best = _channel_sure_batch(h[None], np.asarray([noise_var]))
    return channel[0], SMOOTHER_NAMES[best[0]]


def _channel_sure_batch(h: np.ndarray, noise_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_channel_sure` of the rows of ``h`` [n, active] -> (estimates [n, active], smoother ids [n])."""
    ramp = _channel_ramp(h)
    g = h / ramp
    estimates = (_SMOOTHER_STACK @ g.T).reshape(len(SMOOTHERS), len(ACTIVE), len(g)).transpose(2, 0, 1)  # [n, s, k]
    risk = np.sum(np.abs(estimates - g[:, None]) ** 2, axis=2) + 2 * noise_var[:, None] * _SMOOTHER_TRACE
    best = np.argmin(risk, axis=1)
    return estimates[np.arange(len(h)), best] * ramp, best


def _phase_track(residual: np.ndarray, tracking: str) -> np.ndarray:
    """Phase model theta[t, k] from the pilot residuals (rows = OFDM symbols, cols = PILOT_K).

    "symbol": offset + slope fitted per symbol from its 4 pilots.
    "packet": one offset, one drift per symbol and one slope per carrier for the whole
    packet, from all its pilots (residual CFO and timing offset are linear).
    "hybrid": packet-level slope, per-symbol common phase.
    Returns a function of the carrier indices: coefficients (offset[t], slope[t]).
    """
    if residual.ndim == 3:  # [packets, symbols, pilots]
        if tracking == "symbol":
            return np.stack([_phase_track(r, tracking) for r in residual])
        t = np.arange(residual.shape[1])[:, None]
        slope = np.angle(np.sum(np.conj(residual[..., :-1]) * residual[..., 1:], axis=(1, 2))) / np.diff(PILOT_K)[0]
        derotated = residual * np.exp(-1j * slope[:, None, None] * PILOT_K)
        if tracking == "packet":
            drift = np.angle(np.sum(np.conj(derotated[:, :-1]) * derotated[:, 1:], axis=(1, 2)))
            offset = np.angle(np.sum(derotated * np.exp(-1j * drift[:, None, None] * t), axis=(1, 2)))
            common = offset[:, None] + drift[:, None] * t[:, 0]
        elif tracking == "hybrid":
            common = np.angle(derotated.sum(axis=2))
        else:
            raise ValueError(f"Unknown tracking {tracking!r}")
        return np.stack([common, np.broadcast_to(slope[:, None], common.shape)], axis=2)
    t = np.arange(len(residual))[:, None]
    if tracking == "symbol":
        phases = np.unwrap(np.angle(residual), axis=1)
        design = np.column_stack([np.ones(4), PILOT_K])
        return np.linalg.lstsq(design, phases.T, rcond=None)[0].T
    slope = np.angle(np.sum(np.conj(residual[:, :-1]) * residual[:, 1:])) / np.diff(PILOT_K)[0]
    derotated = residual * np.exp(-1j * slope * PILOT_K)
    if tracking == "packet":
        drift = np.angle(np.sum(np.conj(derotated[:-1]) * derotated[1:]))
        offset = np.angle(np.sum(derotated * np.exp(-1j * drift * t)))
        common = offset + drift * t[:, 0]
    elif tracking == "hybrid":
        common = np.angle(derotated.sum(axis=1))
    else:
        raise ValueError(f"Unknown tracking {tracking!r}")
    return np.column_stack([common, np.full(len(residual), slope)])


def _correlate_rows(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """``signal.correlate(row, template, "valid")`` of every row of ``x``, in one FFT pass."""
    return signal.fftconvolve(x, np.conj(template[::-1])[None], mode="valid", axes=1)


def _stf_candidates(zd: np.ndarray, cross: np.ndarray, peaks: np.ndarray, delay: int) -> list:
    """(score, start, coarse CFO) of every detection peak whose STF correlation passes.

    Each peak's window (-200..+400 samples) is CFO-corrected with the peak's coarse
    estimate and correlated with the STF; windows of full length go through one batch."""
    out = []
    if not len(peaks):
        return out
    coarse = np.angle(cross[peaks]) * BASE_RATE / (2 * np.pi * delay)
    lo, hi = np.maximum(0, peaks - 200), np.minimum(len(zd), peaks + 400)
    stf_energy = np.sum(np.abs(STF) ** 2)

    def scores(local):
        corr = _correlate_rows(local, STF)
        cumulative = np.concatenate([np.zeros((len(local), 1)), np.cumsum(np.abs(local) ** 2, axis=1)], axis=1)
        energy = cumulative[:, len(STF):] - cumulative[:, :-len(STF)]
        return np.abs(corr) ** 2 / np.maximum(stf_energy * energy, 1e-20)

    full = np.flatnonzero(hi - lo == 600)
    results = {}
    if len(full):
        n = np.arange(600)
        local = zd[lo[full, None] + n] * np.exp(-2j * np.pi * coarse[full, None] * n / BASE_RATE)
        for i, row in zip(full, scores(local)):
            results[i] = row
    for i in np.flatnonzero((hi - lo != 600) & (hi - lo >= len(STF))):  # windows cut by the capture's ends
        n = np.arange(hi[i] - lo[i])
        results[i] = scores((zd[lo[i]:hi[i]] * np.exp(-2j * np.pi * coarse[i] * n / BASE_RATE))[None])[0]
    for i in sorted(results):
        index = int(np.argmax(results[i]))
        if results[i][index] > STF_SCORE_THRESHOLD:
            out.append((float(results[i][index]), int(lo[i]) + index, float(coarse[i])))
    return out


_TRACKING_IDS = {"hybrid": 0, "packet": 1, "symbol": 2}
_CHANNEL_IDS = {"sure": 0, "smooth": 1, "ls": 2}


def recover(z: np.ndarray, total_packets: int, seq: int | None, tracking: str = "hybrid",
            channel_method: str = "sure", smooth: int = 7, native: bool | None = None, infer_ids: bool = False):
    """Blind packet recovery from baseband at ``BASE_RATE``.

    ``channel_method``: "sure" (smoother chosen per packet by its risk estimate, default),
    "smooth" (moving average over ``smooth`` carriers, the 1 MS/s receiver) or "ls".
    Returns ``(packets, report)``; ``packets`` maps packet id -> (equalized
    32x48 payload rows, per-packet info). Only CRC-validated headers of the
    expected transmission (``seq``, ``total_packets``) are accepted. All candidate
    packets of the capture are synchronized, equalized and Viterbi-decoded as one batch.
    ``native`` (default: when it compiles) runs everything up to the header CRC check in
    C++ (``ofdm_native.cpp``); its values agree with numpy to rounding.
    ``infer_ids``: also keep packets whose header failed the CRC but whose position and decoded
    header bits identify them (:func:`_infer_packet_ids`); their info has ``"header_crc": False``.
    The payload is still used (analog JSCC values degrade gracefully instead of becoming zeros).
    """
    if tracking not in _TRACKING_IDS:
        raise ValueError(f"Unknown tracking {tracking!r}")
    if channel_method not in _CHANNEL_IDS:
        raise ValueError(f"Unknown channel method {channel_method!r}")
    lib = native_kernel() if native is None or native else None
    if native and lib is None:
        raise RuntimeError("The C++ OFDM kernel is not available.")
    if lib is not None:
        return _recover_native(lib, z, total_packets, seq, tracking, channel_method, smooth, infer_ids)
    z = np.asarray(z, np.complex128).copy()
    dc = complex(np.mean(z))
    z -= dc  # the RX DC offset / LO leakage (the DC carrier is null)
    zd = detection_signal(z)
    delay, window = 16, STF_WINDOW
    cross = _moving_sum(np.conj(zd[:-delay]) * zd[delay:], window)
    e1 = _moving_sum(np.abs(zd[:-delay]) ** 2, window)
    e2 = _moving_sum(np.abs(zd[delay:]) ** 2, window)
    metric = np.abs(cross) ** 2 / np.maximum(e1 * e2, 1e-20)
    peaks, _ = signal.find_peaks(metric, height=STF_THRESHOLD, distance=120)
    candidates = _stf_candidates(zd, cross, peaks, delay)
    chosen = []
    for quality, start, coarse in sorted(candidates, reverse=True):
        if all(abs(start - other[1]) > 100 for other in chosen):
            chosen.append((quality, start, coarse))
    chosen.sort(key=lambda row: row[1])
    rejected = sum(start + PACKET_LEN + 64 >= len(z) for _, start, _ in chosen)
    chosen = [row for row in chosen if row[1] + PACKET_LEN + 64 < len(z)]
    found, duplicates = {}, 0
    report = lambda: {"expected_packets": total_packets, "decoded_packets": len(found),  # noqa: E731
                      "missing_packet_ids": sorted(set(range(total_packets)) - set(found)),
                      "rejected_candidates": rejected, "duplicate_detections": duplicates,
                      "removed_dc": abs(dc)}
    if not chosen:
        return found, report()
    quality = np.array([row[0] for row in chosen])
    starts = np.array([row[1] for row in chosen])
    coarse = np.array([row[2] for row in chosen])

    # LTF timing: correlate 20 samples either side of the expected LTF start
    n = np.arange(len(TRAIN) - len(STF) + 40)
    lo = starts + len(STF) - 20
    local = z[lo[:, None] + n] * np.exp(-2j * np.pi * coarse[:, None] * n / BASE_RATE)
    starts = lo + np.argmax(np.abs(_correlate_rows(local, TRAIN[len(STF):])), axis=1) - len(STF)
    keep = starts >= 0
    quality, starts, coarse = quality[keep], starts[keep], coarse[keep]
    if not len(starts):
        return found, report()

    # fine CFO from the two LTFs, LS channel, noise from their difference
    n = np.arange(PACKET_LEN + 32)
    packets = z[starts[:, None] + n]
    times = n / BASE_RATE
    first = len(STF) + 32 - ADVANCE
    span = slice(first, first + 2 * NFFT)  # the two LTFs, coarse-corrected only where they are used
    ltfs = packets[:, span] * np.exp(-2j * np.pi * coarse[:, None] * times[span])
    l1, l2 = ltfs[:, :NFFT], ltfs[:, NFFT:]
    fine = np.angle(np.sum(np.conj(l1) * l2, axis=1)) * BASE_RATE / (2 * np.pi * NFFT)
    cfo = coarse + fine
    packets = packets * np.exp(-2j * np.pi * cfo[:, None] * times)
    tf = np.fft.fft(packets[:, first:first + 2 * NFFT].reshape(-1, 2, NFFT), axis=2, norm="ortho")
    ltf_noise = np.mean(np.abs(tf[:, 0, ACTIVE_IDX] - tf[:, 1, ACTIVE_IDX]) ** 2, axis=1) / 2
    ls = tf[:, :, ACTIVE_IDX].mean(axis=1) / LTF_GRID[ACTIVE_IDX]
    channel = np.zeros((len(starts), NFFT), complex)
    if channel_method == "sure":
        channel[:, ACTIVE_IDX], best = _channel_sure_batch(ls, ltf_noise / 2)
        smoothers = [SMOOTHER_NAMES[b] for b in best]
    elif channel_method == "smooth":
        channel[:, ACTIVE_IDX] = np.stack([_smooth_channel(h, smooth) for h in ls])
        smoothers = [channel_method] * len(starts)
    elif channel_method == "ls":
        channel[:, ACTIVE_IDX] = ls
        smoothers = [channel_method] * len(starts)
    else:
        raise ValueError(f"Unknown channel method {channel_method!r}")
    usable = np.min(np.abs(channel[:, ACTIVE_IDX]), axis=1) >= 1e-8
    rejected += int(np.sum(~usable))

    # equalize, track the pilot phase, decode every header in one Viterbi pass
    indices = len(TRAIN) + np.arange(HEADER_ROWS + ROWS)[:, None] * SYMLEN + CP - ADVANCE + np.arange(NFFT)
    received = np.fft.fft(packets[:, indices], axis=2, norm="ortho")
    with np.errstate(divide="ignore", invalid="ignore"):  # unusable channels are skipped below
        full = np.zeros_like(received)
        full[:, :, ACTIVE_IDX] = received[:, :, ACTIVE_IDX] / channel[:, None, ACTIVE_IDX]
        residual = full[:, :, PILOT_IDX] / PILOTS
        coefficients = _phase_track(np.where(usable[:, None, None], residual, 1.0), tracking)
        correction = np.exp(1j * (coefficients[:, :, 0, None] + coefficients[:, :, 1, None] * DATA_K))
        data = full[:, :, DATA_IDX] / correction
        gains = np.abs(channel[:, DATA_IDX]) ** 2
        weights = gains / gains.mean(axis=1, keepdims=True)
    decoded = viterbi(_header_soft(np.where(usable[:, None, None], data, 0.0), weights))
    parsed = [_check_header(bits, total_packets, seq) if ok else None for bits, ok in zip(decoded, usable)]
    rejected += sum(p is None for p, ok in zip(parsed, usable) if ok)
    inferred = _infer_packet_ids(parsed, usable, starts, quality, decoded, total_packets, seq) if infer_ids else {}
    accepted = [i for i, p in enumerate(parsed) if p is not None]
    if not accepted:
        return found, report()

    # CRC-validated, re-encoded header symbols: payload-independent SNR at the equalizer output.
    expected = header_symbols([parsed[i][1] for i in accepted], [parsed[i][0] for i in accepted], total_packets)
    row_errors = np.mean(np.abs(data[accepted, :HEADER_ROWS] - expected) ** 2, axis=2)
    ltf_signal = np.mean(np.abs(channel[:, ACTIVE_IDX]) ** 2, axis=1) - ltf_noise / 2
    pilot_fit = np.exp(1j * (coefficients[:, :, 0, None] + coefficients[:, :, 1, None] * PILOT_K))
    # noise in the guard before each packet (CFO-corrected spectrum on the active carriers)
    noise = {}
    whole = [i for i in accepted if starts[i] >= 448]  # full 384-sample guards: one batch
    if whole:
        n = np.arange(384)
        guards = z[starts[whole, None] - 448 + n] * np.exp(-2j * np.pi * cfo[whole, None] * n / BASE_RATE)
        spectrum = np.fft.fft(guards.reshape(len(whole), -1, NFFT), axis=2, norm="ortho")
        noise.update(zip(whole, np.mean(np.abs(spectrum[:, :, ACTIVE_IDX]) ** 2, axis=(1, 2)).tolist()))
    for i in accepted:
        start = int(starts[i])
        guard = z[max(0, start - 448):max(0, start - 64)]
        if i not in noise and len(guard) >= NFFT:
            guard = guard[:len(guard) // NFFT * NFFT] * np.exp(
                -2j * np.pi * float(cfo[i]) * np.arange(len(guard) // NFFT * NFFT) / BASE_RATE)
            spectrum = np.fft.fft(guard.reshape(-1, NFFT), axis=1, norm="ortho")
            noise[i] = float(np.mean(np.abs(spectrum[:, ACTIVE_IDX]) ** 2))
    for j, i in enumerate(accepted):
        packet_id, _ = parsed[i]
        start, cfo_i = int(starts[i]), float(cfo[i])
        noise_power = noise.get(i)
        guard_snr = None
        if noise_power is not None and ltf_signal[i] > 0:
            guard_snr = float(10 * np.log10(ltf_signal[i] / max(noise_power, 1e-20)))
        header_mse = float(np.mean(row_errors[j]))
        magnitude = np.abs(channel[i, ACTIVE_IDX])
        info = {"packet_id": packet_id, "start_baseband_sample": start, "stf_score": float(quality[i]),
                "coarse_cfo_hz": float(coarse[i]), "fine_cfo_hz": float(fine[i]), "total_cfo_hz": cfo_i,
                "guard_noise_power": noise_power, "guard_based_ltf_snr_db": guard_snr,
                "ltf_signal_power": float(ltf_signal[i]),
                "ltf_repeat_snr_db": float(10 * np.log10(ltf_signal[i] / max(ltf_noise[i], 1e-20)))
                if ltf_signal[i] > 0 else None,
                "header_error_power": header_mse, "header_row_errors": row_errors[j].tolist(),
                "header_snr_db": float(-10 * np.log10(max(header_mse, 1e-20))),
                "pilot_residual_evm_percent": float(100 * np.sqrt(np.mean(np.abs(residual[i] - pilot_fit[i]) ** 2))),
                "min_channel_magnitude": float(magnitude.min()),
                "max_channel_magnitude": float(magnitude.max()),
                "channel_power_spread_db": float(10 * np.log10(gains[i].max() / gains[i].min())),
                "channel_smoother": smoothers[i]}
        if packet_id in found:
            duplicates += 1
        if packet_id not in found or info["stf_score"] > found[packet_id][1]["stf_score"]:
            found[packet_id] = (data[i, HEADER_ROWS:], info)
    _add_inferred(found, inferred, data, starts, quality, cfo, total_packets)
    out = report()
    out["inferred_packets"] = sum(1 for _, info in found.values() if info.get("header_crc") is False)
    return found, out


def _add_inferred(found: dict, inferred: dict, data, starts, quality, cfo, total: int) -> None:
    """Header-failed packets identified by :func:`_infer_packet_ids`; header errors against the
    header they should carry, so :func:`snr_estimate` covers them like the others."""
    for i, (packet_id, seq) in inferred.items():
        if packet_id in found:
            continue
        row_errors = np.mean(np.abs(data[i, :HEADER_ROWS] - header_symbols(seq, packet_id, total)) ** 2, axis=1)
        header_mse = float(np.mean(row_errors))
        found[packet_id] = (data[i, HEADER_ROWS:], {
            "packet_id": packet_id, "start_baseband_sample": int(starts[i]), "stf_score": float(quality[i]),
            "total_cfo_hz": float(cfo[i]), "header_crc": False, "header_error_power": header_mse,
            "header_row_errors": row_errors.tolist(), "header_snr_db": float(-10 * np.log10(max(header_mse, 1e-20)))})


def _recover_native(lib, z, total_packets: int, seq: int | None, tracking: str, channel_method: str, smooth: int,
                    infer_ids: bool = False):
    """:func:`recover` with the C++ kernel; the CRC check and the report stay here."""
    z = np.ascontiguousarray(z, np.complex128)
    capacity = 32
    while True:
        dc, rejected = np.zeros(2), np.zeros(1, np.int64)
        starts, quality, coarse, fine, noise = (np.zeros(capacity, t) for t in (np.int64, *[np.float64] * 4))
        channel = np.zeros((capacity, len(ACTIVE)), complex)
        smoother, usable = np.zeros(capacity, np.int32), np.zeros(capacity, np.uint8)
        data = np.zeros((capacity, HEADER_ROWS + ROWS, 48), complex)
        bits = np.zeros((capacity, CODED_BITS // 2), np.uint8)
        stats = np.zeros((capacity, 6))
        count = lib.ofdm_rx_recover(z.reshape(-1).view(np.float64), len(z), _CHANNEL_IDS[channel_method], int(smooth),
                                    _TRACKING_IDS[tracking], capacity, dc, rejected, starts, quality, coarse, fine,
                                    noise, channel.reshape(-1).view(np.float64), smoother, usable,
                                    data.reshape(-1).view(np.float64), bits, stats)
        if count < 0:
            raise RuntimeError("The C++ receiver was not set up.")
        if count <= capacity:
            break
        capacity = int(count)
    found, duplicates = {}, 0
    parsed = [_check_header(bits[i], total_packets, seq) if usable[i] else None for i in range(count)]
    accepted = [i for i, p in enumerate(parsed) if p is not None]
    rejected = int(rejected[0]) + count - len(accepted)
    inferred = _infer_packet_ids(parsed, usable[:count], starts[:count], quality[:count], bits[:count],
                                 total_packets, seq) if infer_ids else {}
    if accepted:
        expected = header_symbols([parsed[i][1] for i in accepted], [parsed[i][0] for i in accepted], total_packets)
        row_errors = np.mean(np.abs(data[accepted, :HEADER_ROWS] - expected) ** 2, axis=2).tolist()
        values = {name: array[accepted].tolist() for name, array in
                  (("start", starts), ("quality", quality), ("coarse", coarse), ("fine", fine), ("noise", noise),
                   ("smoother", smoother), ("stats", stats))}
        for j, i in enumerate(accepted):
            packet_id = parsed[i][0]
            guard_noise, ltf_signal, evm, hmin, hmax, spread = values["stats"][j]
            ltf_noise, header_mse = values["noise"][j], float(np.mean(row_errors[j]))
            noise_power = None if np.isnan(guard_noise) else guard_noise
            name = SMOOTHER_NAMES[values["smoother"][j]] if channel_method == "sure" else channel_method
            info = {"packet_id": packet_id, "start_baseband_sample": values["start"][j],
                    "stf_score": values["quality"][j], "coarse_cfo_hz": values["coarse"][j],
                    "fine_cfo_hz": values["fine"][j], "total_cfo_hz": values["coarse"][j] + values["fine"][j],
                    "guard_noise_power": noise_power,
                    "guard_based_ltf_snr_db": float(10 * np.log10(ltf_signal / max(noise_power, 1e-20)))
                    if noise_power is not None and ltf_signal > 0 else None,
                    "ltf_signal_power": ltf_signal,
                    "ltf_repeat_snr_db": float(10 * np.log10(ltf_signal / max(ltf_noise, 1e-20)))
                    if ltf_signal > 0 else None,
                    "header_error_power": header_mse, "header_row_errors": row_errors[j],
                    "header_snr_db": float(-10 * np.log10(max(header_mse, 1e-20))),
                    "pilot_residual_evm_percent": evm, "min_channel_magnitude": hmin,
                    "max_channel_magnitude": hmax, "channel_power_spread_db": spread, "channel_smoother": name}
            if packet_id in found:
                duplicates += 1
            if packet_id not in found or info["stf_score"] > found[packet_id][1]["stf_score"]:
                found[packet_id] = (data[i, HEADER_ROWS:], info)
    _add_inferred(found, inferred, data, starts, quality, coarse + fine, total_packets)
    report = {"expected_packets": total_packets, "decoded_packets": len(found),
              "missing_packet_ids": sorted(set(range(total_packets)) - set(found)),
              "rejected_candidates": rejected, "duplicate_detections": duplicates,
              "inferred_packets": sum(1 for _, info in found.values() if info.get("header_crc") is False),
              "removed_dc": abs(complex(dc[0], dc[1]))}
    return found, report


def snr_estimate(infos) -> float | None:
    """Model-condition SNR: -10 log10 of the median equalized error power of the header OFDM
    symbols of the given packets (4 per packet). The median over symbols keeps short
    interference bursts, which hit single 8 us symbols, from dragging the estimate down;
    for Gaussian noise it equals the mean within 0.03 dB (48 symbols per row)."""
    errors = [e for info in infos for e in info["header_row_errors"]]
    return float(-10 * np.log10(max(float(np.median(errors)), 1e-20))) if errors else None


def probe_snr(packets: dict, seed: int) -> float | None:
    """SNR from the known payload of a probe (:func:`probe_rows`): -10 log10 of the median
    equalized error power per OFDM symbol (64 symbols of 48 carriers)."""
    if not packets:
        return None
    rows = probe_rows(seed)
    errors = np.concatenate([np.mean(np.abs(data - rows[p * ROWS:(p + 1) * ROWS]) ** 2, axis=1)
                             for p, (data, _) in packets.items()])
    return float(-10 * np.log10(max(float(np.median(errors)), 1e-20)))


def assemble(packets: dict, total_packets: int, perm: np.ndarray, scramble_seed: int | None = None) -> np.ndarray:
    """Packet dict -> data rows [total*ROWS, 48] in feature order; missing packets are NaN."""
    rows = np.full((total_packets * ROWS, 48), np.nan + 1j * np.nan)
    for packet_id, (data, _) in packets.items():
        rows[packet_id * ROWS:(packet_id + 1) * ROWS] = data
    rows = rows * np.conj(scrambling(len(rows), scramble_seed))
    return permute_rows(rows, perm, inverse=True)


# ---------------------------------------------------------------- simulation

def simulate_channel(iq: np.ndarray, *, cfo_hz: float = 0.0, delay: int = 0, phase: float = 0.0,
                     taps=(1.0,), noise_counts: float = 0.0, lead_seconds: float = 0.005,
                     spur=0.0, tone=0.0, seed: int = 0) -> np.ndarray:
    """Software stand-in for the air link at ``RATE`` (int8 I/Q in and out, like a capture).

    ``spur``: constant at the receiver's DC; ``tone``: complex amplitude of the TX LO
    leakage, at the TX band centre (it goes through the channel and the CFO and is
    present before, between and after the packets, as with a continuous TX stream).
    """
    rng = np.random.default_rng(seed)
    x = iq[:, 0].astype(float) + 1j * iq[:, 1]
    x = np.r_[np.zeros(int(lead_seconds * RATE) + delay), x, np.zeros(int(lead_seconds * RATE))] + tone
    x = np.convolve(x, np.asarray(taps, complex))[:len(x)]
    n = np.arange(len(x))
    x = x * np.exp(1j * (2 * np.pi * cfo_hz * n / RATE + phase))
    x = x + spur + noise_counts * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) / np.sqrt(2)
    out = np.empty((len(x), 2), np.int8)
    out[:, 0] = np.clip(np.rint(x.real), -128, 127)
    out[:, 1] = np.clip(np.rint(x.imag), -128, 127)
    return out
