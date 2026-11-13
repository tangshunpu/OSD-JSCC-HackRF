"""HackRF One I/O through ``hackrf_transfer``, plus the TCP control messages of the RF link.

- :class:`RxStream` keeps one receiver running for the whole session
  (``hackrf_transfer -r -``) and copies its samples into a ring buffer, so a
  transmission is cut out of the stream by sample position; the receiver
  is never restarted between images.
- :class:`TxStream` keeps one transmitter running (``hackrf_transfer -t -``):
  a separate process feeds it zeros and splices each burst into the sample
  stream, so there is no process start-up or settling time per burst, and the
  burst's position in the TX sample stream is known exactly.
- :class:`NativeTxStream` is the same stream through libhackrf in this process
  (``hackrf_tx_native.cpp``) with short USB transfers, so a burst waits behind
  ~4 ms of queued samples instead of ~26 ms (``--tx-transfer-kib``).
- :func:`transmit` sends one finite burst (``-n`` samples, no repeat) and exits.
  RF amplifier and antenna bias stay off on both ends (``amp=True``: TX RF amplifier on, +~11 dB).
- :func:`send_msg` / :func:`recv_msg` frame the control channel: a JSON
  object plus an optional binary blob. Only control data and (after decoding,
  for scoring only) the clean sender symbols travel over TCP; the JSCC payload
  goes over the air.
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import time

import numpy as np

import ofdm_phy

RATE = ofdm_phy.RATE         # 20 MS/s
BANDWIDTH = 12_000_000       # MAX2837 baseband filter (two-sided); the band spans +-4.06 MHz
MAX_TX_SECONDS = 2.0
MAX_TX_GAIN = 47
_FRAME = struct.Struct("<II")
_F_SETPIPE_SZ = 1031


def hackrf_transfer() -> str:
    path = os.environ.get("HACKRF_TRANSFER") or shutil.which("hackrf_transfer")
    if not path:
        raise SystemExit("hackrf_transfer not found (install the 'hackrf' package or set HACKRF_TRANSFER).")
    return path


def common_args(lo_hz: int) -> list[str]:
    # -a 0: RF amplifier off, -p 0: antenna port power off, -B: buffer/overrun statistics.
    return ["-f", str(int(lo_hz)), "-s", str(RATE), "-b", str(BANDWIDTH), "-a", "0", "-p", "0", "-B"]


def _counts(text: str, word: str) -> list[int]:
    return [int(n) for n in re.findall(rf"(\d+) {word}", text)]


def _rx_reader(command: list[str], shm_name: str, size: int, counter, stop, log_path: str) -> None:
    """Child process: hackrf_transfer stdout -> shared ring buffer; owns no CUDA or GIL contention."""
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=shm_name)
    view = shm.buf
    with open(log_path, "w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, bufsize=0)
        try:  # 1 MiB pipe (~65 ms of samples) absorbs scheduling stalls
            fcntl.fcntl(proc.stdout.fileno(), _F_SETPIPE_SZ, 1 << 20)
        except OSError:
            pass
        total = 0
        try:
            while not stop.is_set():
                pos = total % size
                n = proc.stdout.readinto(view[pos:min(pos + RxStream.CHUNK, size)])
                if not n:
                    break
                total += n
                with counter.get_lock():
                    counter.value = total
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            del view
            shm.close()


class RxStream:
    """Continuous capture into a ring buffer of ``seconds`` of int8 I/Q.

    A forked child process drains ``hackrf_transfer -r -`` into shared memory,
    so model loading or decoding in the main process cannot stall the pipe
    (a stalled pipe makes the HackRF drop samples). Create it before CUDA is
    initialized in the parent.
    """

    CHUNK = 1 << 18

    def __init__(self, serial: str, lo_hz: int, lna: int, vga: int, log_path: Path,
                 seconds: float = 8.0) -> None:
        import multiprocessing as mp
        from multiprocessing import shared_memory

        if lna not in range(0, 41, 8) or vga not in range(0, 63, 2):
            raise ValueError("LNA gain must be 0-40 in steps of 8, VGA gain 0-62 in steps of 2.")
        self.size = int(seconds * RATE) * 2
        self.shm = shared_memory.SharedMemory(create=True, size=self.size)
        self.buffer = np.ndarray((self.size,), np.int8, self.shm.buf)
        self.log_path = Path(log_path)
        self.command = [hackrf_transfer(), "-d", serial, "-r", "-", "-l", str(lna), "-g", str(vga),
                        *common_args(lo_hz)]
        ctx = mp.get_context("fork")
        self.counter, self.stop = ctx.Value("q", 0), ctx.Event()
        self.process = ctx.Process(target=_rx_reader, daemon=True, args=(
            self.command, self.shm.name, self.size, self.counter, self.stop, str(self.log_path)))
        self.process.start()

    def _total(self) -> int:
        with self.counter.get_lock():
            return self.counter.value

    def position(self) -> int:
        """Complex samples received so far."""
        return self._total() // 2

    def wait_until(self, sample: int, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self._total() // 2 < sample:
            if not self.process.is_alive():
                raise RuntimeError(f"RX stream ended; see {self.log_path}")
            if time.monotonic() > deadline:
                raise TimeoutError("RX stream stalled.")
            time.sleep(0.001)

    def wait_started(self, seconds: float = 0.2, timeout: float = 5.0) -> None:
        self.wait_until(int(seconds * RATE), timeout)

    def read(self, start: int, end: int) -> np.ndarray:
        """Copy samples [start, end) as int8 [n, 2]; raises if the ring already overwrote them."""
        total = self._total()
        if not 0 <= start <= end or 2 * end > total:
            raise ValueError(f"Samples {start}:{end} not captured yet (have {total // 2}).")
        if 2 * start < total - self.size:
            raise ValueError("Requested samples were overwritten; enlarge the ring buffer.")
        a, b = (2 * start) % self.size, (2 * end) % self.size
        if end == start:
            out = np.empty(0, np.int8)
        elif a < b:
            out = self.buffer[a:b].copy()
        else:
            out = np.concatenate([self.buffer[a:], self.buffer[:b]])
        if 2 * start < self._total() - self.size:  # overwritten while copying
            raise ValueError("Requested samples were overwritten during the copy.")
        return out.reshape(-1, 2)

    def overruns(self) -> int:
        """hackrf_transfer's cumulative overrun counter (updated once per second)."""
        counts = _counts(self.log_path.read_text(errors="replace"), "overruns")
        return counts[-1] if counts else 0

    def close(self) -> None:
        self.stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        del self.buffer
        self.shm.close()
        self.shm.unlink()


def _tx_feeder(command: list[str], shm_name: str, conn, counter, stop, log_path: str, cores) -> None:
    """Child process: zeros -> hackrf_transfer stdin, with each requested burst spliced in.

    The 64 KiB pipe is kept full, so writes are paced by the HackRF; a burst starts at
    the TX stream sample ``counter / 2`` reported back to the parent. With ``cores``, the
    feeder and hackrf_transfer run only there (the parent keeps the other cores).
    """
    from multiprocessing import shared_memory

    if cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)  # inherited by hackrf_transfer
    shm = shared_memory.SharedMemory(name=shm_name)
    view = shm.buf
    zeros = bytes(TxStream.CHUNK)
    with open(log_path, "a") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log, bufsize=0)
        fd = proc.stdin.fileno()
        try:
            fcntl.fcntl(fd, _F_SETPIPE_SZ, TxStream.CHUNK)
        except OSError:
            pass
        written = 0
        try:
            while not stop.is_set():
                if conn.poll():
                    size = conn.recv()
                    start, t_start = written, time.time_ns()
                    data = view[:size]
                    sent = 0
                    while sent < size:
                        sent += os.write(fd, data[sent:])
                    del data
                    written += size
                    conn.send({"tx_stream_sample": start // 2, "t_write_start": t_start, "t_write_end": time.time_ns()})
                written += os.write(fd, zeros)
                counter.value = written
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            del view
            shm.close()


class TxStream:
    """Continuous transmitter at one TX VGA gain; :meth:`send` splices a burst into the stream.

    The stream radiates zeros between bursts (the LO leakage stays on the air, at the
    band centre). A gain change needs a new stream. ``stream_id`` changes with every
    stream, so the receiver knows when the TX sample counter restarted.
    """

    CHUNK = 1 << 16
    MAX_BURST_SECONDS = 0.2

    def __init__(self, serial: str, lo_hz: int, gain: int, log_path: Path, cores=(0, 1), amp: bool = False) -> None:
        """``cores``: CPUs reserved for the feeder and hackrf_transfer; this process moves to the
        others, so encoders (TensorRT host threads, Kvazaar) cannot starve the stream (the AGX
        has no real-time priority without root). Empty: no pinning."""
        import multiprocessing as mp
        from multiprocessing import shared_memory

        if not 0 <= gain <= MAX_TX_GAIN:
            raise ValueError(f"TX VGA gain must be 0-{MAX_TX_GAIN} dB.")
        self.cores = set(cores or ()) if hasattr(os, "sched_setaffinity") else set()  # no pinning on macOS
        if self.cores:
            others = set(os.sched_getaffinity(0)) - self.cores
            if not others:
                raise ValueError("No CPU left for the sender after reserving the TX cores.")
            os.sched_setaffinity(0, others)
        self.gain, self.log_path = gain, Path(log_path)
        self.stream_id = time.time_ns() // 1000 % (1 << 31)
        self.command = [hackrf_transfer(), "-d", serial, "-t", "-", "-x", str(gain), *common_args(lo_hz)]
        if amp:
            self.command[self.command.index("-a") + 1] = "1"
        self.size = int(self.MAX_BURST_SECONDS * RATE) * 2
        self.shm = shared_memory.SharedMemory(create=True, size=self.size)
        ctx = mp.get_context("spawn")  # the parent may already hold CUDA/TensorRT state
        self.conn, child = ctx.Pipe()
        self.counter, self.stop = ctx.Value("q", 0, lock=False), ctx.Event()
        self.process = ctx.Process(target=_tx_feeder, daemon=True, args=(
            self.command, self.shm.name, child, self.counter, self.stop, str(self.log_path), self.cores))
        self.process.start()
        deadline = time.monotonic() + 10
        while self.counter.value < int(0.3 * RATE) * 2:  # streaming for 0.3 s
            if not self.process.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"TX stream did not start; see {self.log_path}")
            time.sleep(0.01)

    def send(self, iq: np.ndarray) -> dict:
        """Queue one int8 [n, 2] burst; returns once it is in the pipe (TX stream position, times)."""
        data = np.ascontiguousarray(iq, np.int8).reshape(-1)
        if len(data) > self.size:
            raise ValueError(f"Burst longer than {self.MAX_BURST_SECONDS} s.")
        np.ndarray((len(data),), np.int8, self.shm.buf)[:] = data
        t_queue = time.time_ns()
        self.conn.send(len(data))
        if not self.conn.poll(5):
            raise RuntimeError(f"TX feeder stalled; see {self.log_path}")
        reply = self.conn.recv()
        return {**reply, "t_tx_queue": t_queue, "tx_stream_id": self.stream_id, "tx_gain": self.gain,
                "underruns": self.underruns(), "returncode": 0 if self.process.is_alive() else 1,
                "tx_backend": "hackrf_transfer"}

    def underruns(self) -> int:
        """hackrf_transfer's cumulative underrun counter (updated once per second)."""
        try:
            counts = _counts(self.log_path.read_text(errors="replace").rsplit("$ ", 1)[-1], "underruns")
        except OSError:
            return 0
        return counts[-1] if counts else 0

    def close(self) -> None:
        self.stop.set()
        self.process.join(timeout=8)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:  # the feeder's resource tracker may have unlinked it already
            pass


_NATIVE_TX_SOURCE = Path(__file__).with_name("hackrf_tx_native.cpp")
_native_tx = None
_TX_STATS = ("queued", "completed", "start_byte", "end_byte", "t_start", "t_end", "t_accepted", "started_seq",
             "done_seq", "accepted_seq", "shortfalls", "longest_shortfall", "streaming", "callbacks",
             "max_callback_ns", "failed_transfers", "polls", "poll_errors", "transfer_bytes")


def native_tx_lib():
    """The libhackrf TX streamer (``hackrf_tx_native.cpp``), compiled on first use against the
    installed libhackrf into ``$OFDM_NATIVE_DIR`` (default ``~/.cache/osd_ofdm``) under the hash
    of its source. ``HACKRF_LDFLAGS`` overrides the link flags."""
    global _native_tx
    if _native_tx is None:
        digest = hashlib.sha1(_NATIVE_TX_SOURCE.read_bytes()).hexdigest()[:12]
        cache = Path(os.environ.get("OFDM_NATIVE_DIR", Path.home() / ".cache" / "osd_ofdm"))
        stem = f"hackrf_tx_native_{{}}_{platform.system()}_{platform.machine()}.so"
        path = cache / stem.format(digest)
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            base = [os.environ.get("CXX", "c++"), "-O2", "-std=c++17", "-shared", "-fPIC", "-pthread",
                    "-o", str(tmp), str(_NATIVE_TX_SOURCE)]
            if os.environ.get("HACKRF_LDFLAGS"):
                options = [os.environ["HACKRF_LDFLAGS"].split()]
            elif platform.system() == "Darwin":
                options = [["-L/opt/homebrew/lib", "-L/usr/local/lib", "-lhackrf"]]
            else:  # Debian/Ubuntu ship only the runtime libhackrf.so.0 without the -dev package
                options = [["-l:libhackrf.so.0"], ["-lhackrf"]]
            errors = []
            for flags in options:
                result = subprocess.run(base + flags, capture_output=True, text=True)
                if result.returncode == 0:
                    break
                errors.append(result.stderr.strip())
            else:
                raise RuntimeError("Cannot build hackrf_tx_native.cpp against libhackrf: " + " | ".join(errors))
            os.replace(tmp, path)
            for stale in cache.glob(stem.format("*")):
                if stale != path:  # builds of earlier versions of the source
                    stale.unlink(missing_ok=True)
        lib = ctypes.CDLL(str(path))
        cint, ci64 = ctypes.c_int, ctypes.c_int64
        lib.txn_library_release.restype = ctypes.c_char_p
        lib.txn_open.restype = cint
        lib.txn_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_double, ctypes.c_uint32, cint, cint,
                                 cint, cint, ci64, ctypes.POINTER(cint), cint, ctypes.c_char_p, cint]
        lib.txn_burst_buffer.restype = ctypes.c_void_p
        lib.txn_submit.restype, lib.txn_submit.argtypes = cint, [ci64, ci64]
        lib.txn_wait.restype, lib.txn_wait.argtypes = cint, [ci64, cint, cint]
        lib.txn_stats.restype = None
        lib.txn_stats.argtypes = [np.ctypeslib.ndpointer(np.int64, flags="C")]
        lib.txn_close.restype = cint
        _native_tx = lib
    return _native_tx


class NativeTxStream:
    """Continuous transmitter through libhackrf in this process (``hackrf_tx_native.cpp``).

    Same interface and TX-stream sample counter as :class:`TxStream`, but the USB transfers
    are ``transfer_kib`` KiB instead of hackrf_transfer's 256 KiB and there is no pipe, so a
    burst waits behind 4 short transfers and the 32 KiB device buffer (32 KiB: ~4 ms at
    20 MS/s) instead of ~1 MiB of queued zeros (~26 ms). The queue is also the only slack
    against host stalls: a stall longer than it underruns the radio (``underruns()``).
    A burst starts at a transfer boundary. libhackrf's transfer thread (it runs the TX
    callback, in C) is pinned to ``cores``; this process moves to the other CPUs.
    """

    MAX_BURST_SECONDS = TxStream.MAX_BURST_SECONDS

    def __init__(self, serial: str, lo_hz: int, gain: int, log_path: Path, cores=(0, 1),
                 transfer_kib: float = 32, amp: bool = False) -> None:
        if not 0 <= gain <= MAX_TX_GAIN:
            raise ValueError(f"TX VGA gain must be 0-{MAX_TX_GAIN} dB.")
        self.transfer_bytes = int(round(transfer_kib * 1024))
        if not 512 <= self.transfer_bytes <= 256 * 1024 or self.transfer_bytes % 512:
            raise ValueError("TX transfer size must be a multiple of 0.5 KiB, at most 256 KiB.")
        self.lib = lib = native_tx_lib()
        self.cores = set(cores or ()) if hasattr(os, "sched_setaffinity") else set()  # no pinning on macOS
        if self.cores:
            others = set(os.sched_getaffinity(0)) - self.cores
            if not others:
                raise ValueError("No CPU left for the sender after reserving the TX cores.")
            os.sched_setaffinity(0, others)
        self.gain, self.log_path = gain, Path(log_path)
        self.stream_id = time.time_ns() // 1000 % (1 << 31)
        self.size = int(self.MAX_BURST_SECONDS * RATE) * 2
        release = lib.txn_library_release().decode()
        args = common_args(lo_hz)
        args[args.index("-a") + 1] = "1" if amp else "0"
        self.command = [f"libhackrf-{release}", "-d", serial, "-t", "native", "-x", str(gain), *args,
                        "--transfer-bytes", str(self.transfer_bytes), "--cores", *map(str, sorted(self.cores))]
        cpus = sorted(self.cores)
        error = ctypes.create_string_buffer(512)
        code = lib.txn_open(serial.encode(), int(lo_hz), float(RATE), BANDWIDTH, int(bool(amp)), 0, int(gain),
                            self.transfer_bytes,
                            self.size, (ctypes.c_int * len(cpus))(*cpus) if cpus else None, len(cpus), error,
                            len(error))
        if code:
            raise RuntimeError(f"libhackrf TX stream did not start: {error.value.decode(errors='replace')}")
        self._open = True
        atexit.register(self.close)
        self.buffer = np.ctypeslib.as_array((ctypes.c_int8 * self.size).from_address(lib.txn_burst_buffer()))
        self.seq, self.bursts = 0, []
        with open(self.log_path, "a") as log:
            log.write(f"$ {' '.join(self.command)}\n")
        deadline = time.monotonic() + 10
        while self.stats()["queued"] < int(0.3 * RATE) * 2:  # streaming for 0.3 s
            if not self.stats()["streaming"] or time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"libhackrf TX stream did not start streaming; see {self.log_path}")
            time.sleep(0.01)

    def stats(self) -> dict:
        out = np.zeros(len(_TX_STATS), np.int64)
        self.lib.txn_stats(out)
        return dict(zip(_TX_STATS, (int(v) for v in out)))

    def _record(self, seq: int) -> None:
        """Times of finished burst ``seq`` (start = picked by the callback; end = last byte
        handed to libhackrf; accepted = its last USB transfer completed)."""
        if self.lib.txn_wait(seq, 2, 200) == 0:
            s = self.stats()
            self.bursts.append({k: s[k] for k in ("start_byte", "end_byte", "t_start", "t_end", "t_accepted")})

    def send(self, iq: np.ndarray) -> dict:
        """Queue one int8 [n, 2] burst; returns once the TX callback has started it (TX stream
        position, times)."""
        data = np.ascontiguousarray(iq, np.int8).reshape(-1)
        if len(data) > self.size:
            raise ValueError(f"Burst longer than {self.MAX_BURST_SECONDS} s.")
        if self.seq:
            if self.lib.txn_wait(self.seq, 1, 5000) != 0:
                raise RuntimeError(f"Previous TX burst did not finish; see {self.log_path}")
            self._record(self.seq)
        self.buffer[:len(data)] = data
        self.seq += 1
        t_queue = time.time_ns()
        code = self.lib.txn_submit(len(data), self.seq)
        if code:
            raise RuntimeError(f"txn_submit failed ({code}).")
        waited = self.lib.txn_wait(self.seq, 0, 5000)
        if waited:
            raise RuntimeError(f"libhackrf TX stream {'stopped' if waited == 2 else 'stalled'}; see {self.log_path}")
        s = self.stats()
        return {"tx_stream_sample": s["start_byte"] // 2, "t_write_start": s["t_start"], "t_write_end": s["t_start"],
                "t_tx_queue": t_queue, "tx_stream_id": self.stream_id, "tx_gain": self.gain,
                "underruns": s["shortfalls"], "returncode": 0 if s["streaming"] else 1,
                "tx_backend": "libhackrf", "tx_transfer_bytes": self.transfer_bytes}

    def underruns(self) -> int:
        """Cumulative TX underruns (M0 shortfalls, read every 200 ms)."""
        return self.stats()["shortfalls"]

    def close(self) -> None:
        if not getattr(self, "_open", False):
            return
        self._open = False
        if self.seq:
            self._record(self.seq)
        s = self.stats()
        code = self.lib.txn_close()
        queue_ms = [(b["t_accepted"] - b["t_start"]) / 1e6 - (b["end_byte"] - b["start_byte"]) / 2 / RATE * 1e3
                    for b in self.bursts if b["t_accepted"] > 0]
        with open(self.log_path, "a") as log:
            log.write(f"libhackrf TX stream closed ({code}): {s['queued'] / 2 / RATE:.1f} s streamed, "
                      f"{s['shortfalls']} underruns, longest {s['longest_shortfall']} bytes, "
                      f"{s['failed_transfers']} failed transfers, {s['callbacks']} callbacks "
                      f"(max {s['max_callback_ns'] / 1e3:.0f} us), {len(self.bursts)} bursts, "
                      f"start-to-USB-accept minus burst duration median "
                      f"{float(np.median(queue_ms)) if queue_ms else float('nan'):.2f} ms\n")


def open_tx_stream(serial: str, lo_hz: int, gain: int, log_path: Path, cores=(0, 1), transfer_kib: float = 0,
                   amp: bool = False):
    """:class:`NativeTxStream` with ``transfer_kib``-KiB transfers, or with ``transfer_kib`` 0 (or
    when the native streamer cannot be built) :class:`TxStream` (hackrf_transfer through a pipe,
    4 x 256 KiB transfers). The ``send`` reply names the one used (``tx_backend``)."""
    if transfer_kib:
        try:
            native_tx_lib()
        except (OSError, RuntimeError) as error:  # no compiler/libhackrf: keep the measured old path
            print(f"hackrf_io: native TX stream unavailable ({error}); using hackrf_transfer", flush=True)
        else:
            return NativeTxStream(serial, lo_hz, gain, log_path, cores, transfer_kib, amp)
    return TxStream(serial, lo_hz, gain, log_path, cores, amp)


def transmit(path: Path, serial: str, lo_hz: int, gain: int, samples: int, log_path: Path | None = None) -> dict:
    """One finite TX burst of ``samples`` int8 I/Q pairs from ``path``; returns timing and stats."""
    if not 0 <= gain <= MAX_TX_GAIN:
        raise ValueError(f"TX VGA gain must be 0-{MAX_TX_GAIN} dB.")
    if not 0 < samples <= MAX_TX_SECONDS * RATE or Path(path).stat().st_size != 2 * samples:
        raise ValueError(f"TX file must hold exactly {samples} samples, at most {MAX_TX_SECONDS} s.")
    command = [hackrf_transfer(), "-d", serial, "-t", str(path), "-n", str(samples), "-x", str(gain),
               *common_args(lo_hz)]
    start = time.time_ns()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=10)
    end = time.time_ns()
    log = proc.stdout + proc.stderr
    if log_path is not None:
        with open(log_path, "a") as handle:
            handle.write(f"$ {' '.join(command)}\n{log}\n")
    underruns = _counts(log, "underruns")
    return {"returncode": proc.returncode, "t_tx_start": start, "t_tx_end": end,
            "tx_process_ms": (end - start) / 1e6, "underruns": max(underruns) if underruns else None,
            "tx_log_tail": log.strip().splitlines()[-1] if log.strip() else ""}


# ----------------------------------------------------------- control channel

def send_msg(sock, message: dict, blob: bytes = b"") -> None:
    data = json.dumps(message).encode()
    sock.sendall(_FRAME.pack(len(data), len(blob)) + data + blob)


def _recv_exact(sock, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("Peer closed the control connection.")
        data += chunk
    return bytes(data)


def recv_msg(sock) -> tuple[dict, bytes]:
    size, blob_size = _FRAME.unpack(_recv_exact(sock, _FRAME.size))
    message = json.loads(_recv_exact(sock, size))
    return message, _recv_exact(sock, blob_size) if blob_size else b""


def expect(sock, kind: str) -> tuple[dict, bytes]:
    message, blob = recv_msg(sock)
    if message.get("type") != kind:
        raise RuntimeError(f"Expected '{kind}', got {message}")
    return message, blob


def clock_offset(sock, rounds: int = 50) -> tuple[int, int]:
    """Peer clock minus local clock (ns) from the lowest-RTT exchange, and that RTT."""
    best = None
    for _ in range(rounds):
        t_a = time.time_ns()
        send_msg(sock, {"type": "sync", "t_a": t_a})
        reply, _ = expect(sock, "sync")
        t_c = time.time_ns()
        if best is None or t_c - t_a < best[0]:
            best = (t_c - t_a, reply["t_b"] - (t_a + t_c) // 2)
    return best[1], best[0]
