"""Wire format of the live camera demo (``live_sender.py`` -> ``live_receiver.py``).

Every message is ``MSG`` (4-byte kind, body length) followed by the body.

Sender -> receiver, stream socket (port P):
    HELO  JSON: model, frame size, source, camera settings
    FRAM  FRAME header + fp16 symbols [channels, H/16, W/16] (the layer prefix of the rate)
    PONG  PONG: answer to a PING
    BYE!  empty
Receiver -> sender, same socket:
    PING  PING: receiver clock, for the clock offset
    ACK!  ACK: per-frame receiver timestamps (also the sender's flow control)
    CTRL  JSON: {"snr_db": .., "rate": ..} chosen on the web page
Sender -> receiver, preview socket (port P + 1):
    PREV  PREVIEW header + JPEG of the frame the sender encoded (display only)

All timestamps are ``time.time_ns()`` of the machine that took them. The
receiver estimates its clock minus the sender clock from the PING/PONG
exchange with the lowest round trip.
"""

from __future__ import annotations

import json
import socket
import struct
import threading

MSG = struct.Struct("<4sI")
# seq, camera frame index, t_capture, t_pick, t_encode_start, t_encode_end, snr_db, rate, H, W, channels
FRAME = struct.Struct("<IIqqqqffHHH")
# seq, t_recv, t_decode_start, t_decode_end, t_out, dropped
ACK = struct.Struct("<IqqqqB")
PING = struct.Struct("<q")
PONG = struct.Struct("<qq")
PREVIEW = struct.Struct("<Iq")  # seq, t_capture
DEFAULT_PORT = 50020


class Channel:
    """A TCP socket with locked whole-message sends and buffered reads."""

    def __init__(self, sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock
        self.reader = sock.makefile("rb", buffering=1 << 20)
        self.lock = threading.Lock()

    def send(self, kind: bytes, *parts: bytes) -> None:
        body = b"".join(parts)
        with self.lock:
            self.sock.sendall(MSG.pack(kind, len(body)) + body)

    def send_json(self, kind: bytes, value: dict) -> None:
        self.send(kind, json.dumps(value).encode())

    def recv(self) -> tuple[bytes, bytes]:
        header = self.reader.read(MSG.size)
        if len(header) < MSG.size:
            raise ConnectionError("peer closed the connection")
        kind, size = MSG.unpack(header)
        body = self.reader.read(size) if size else b""
        if len(body) < size:
            raise ConnectionError("peer closed the connection")
        return kind, body

    def close(self) -> None:
        for closer in (lambda: self.sock.shutdown(socket.SHUT_RDWR), self.reader.close, self.sock.close):
            try:
                closer()
            except OSError:
                pass
