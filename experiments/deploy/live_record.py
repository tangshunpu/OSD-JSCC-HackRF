"""Record the live demo dashboard (``live_viewer.html``) to an H.264 mp4.

Started and stopped by ``live_receiver.py`` (Record button on the page): a
headless Chromium on this host opens the dashboard in capture mode (no record
controls), sized to the whole page, and the DevTools screencast hands every
repainted frame (JPEG) to this process. A writer thread samples the newest
frame at ``--fps`` (constant frame rate, wall clock) into ffmpeg (libx264,
static binary from ``imageio-ffmpeg``). Stop with a line ``stop`` on stdin,
EOF, SIGINT or SIGTERM; the last stdout line is a JSON summary.

    uv run --no-project --with playwright --with imageio-ffmpeg \
        python experiments/deploy/live_record.py --url http://127.0.0.1:8777/ --out dash.mp4
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def chromium() -> str | None:
    """Newest cached Playwright headless shell (the pip package may expect another build)."""
    root = Path.home() / ".cache" / "ms-playwright"
    for pattern in ("chromium_headless_shell-*/*/chrome-headless-shell", "chromium-*/*/chrome"):
        found = sorted(glob.glob(str(root / pattern)), key=lambda p: int(Path(p).parts[-3].split("-")[-1]))
        if found:
            return found[-1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8777/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1440, help="CSS px; the height follows the page")
    parser.add_argument("--scale", type=float, default=1.25, help="device pixels per CSS px")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--quality", type=int, default=85, help="screencast JPEG quality")
    parser.add_argument("--nice", type=int, default=10,
                        help="lower CPU priority (inherited by Chromium and ffmpeg) so the receiver keeps its timing")
    args = parser.parse_args()
    if args.nice:
        os.nice(args.nice)

    import imageio_ffmpeg
    from playwright.sync_api import sync_playwright

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def watch_stdin() -> None:
        for line in sys.stdin:
            if line.strip() == "stop":
                break
        stop.set()

    threading.Thread(target=watch_stdin, daemon=True).start()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    latest = {"jpeg": None, "count": 0}
    lock = threading.Lock()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium())
        context = browser.new_context(viewport={"width": args.width, "height": 900},
                                      device_scale_factor=args.scale, color_scheme=args.theme)
        page = context.new_page()
        page.goto(args.url.rstrip("/") + "/?capture=1")
        page.wait_for_timeout(2500)  # streams attached, first stats rendered
        height = int(page.evaluate("document.documentElement.scrollHeight"))
        page.set_viewport_size({"width": args.width, "height": height})
        page.wait_for_timeout(300)
        size = (2 * round(args.width * args.scale / 2), 2 * round(height * args.scale / 2))
        ffmpeg = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
             "-framerate", f"{args.fps:g}", "-i", "-", "-vf", f"scale={size[0]}:{size[1]}", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out)], stdin=subprocess.PIPE)
        cdp = context.new_cdp_session(page)

        def on_frame(params: dict) -> None:
            with lock:
                latest["jpeg"] = base64.b64decode(params["data"])
                latest["count"] += 1
            if not stop.is_set():
                try:
                    cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
                except Exception:  # the page is closing
                    pass

        cdp.on("Page.screencastFrame", on_frame)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": args.quality,
                                          "maxWidth": size[0], "maxHeight": size[1], "everyNthFrame": 1})
        written = [0]

        def writer() -> None:
            start = time.perf_counter()
            while not stop.is_set():
                with lock:
                    jpeg = latest["jpeg"]
                if jpeg is not None:
                    try:
                        ffmpeg.stdin.write(jpeg)
                    except BrokenPipeError:
                        stop.set()
                        return
                    written[0] += 1
                delay = start + written[0] / args.fps - time.perf_counter() if jpeg is not None else 0.01
                if delay > 0:
                    time.sleep(delay)

        thread = threading.Thread(target=writer, daemon=True)
        t0 = time.time()
        thread.start()
        print(json.dumps({"status": "recording", "out": str(out), "size": size}), flush=True)
        while not stop.is_set():
            page.wait_for_timeout(50)  # lets Playwright dispatch the screencast events
        thread.join(timeout=2)
        seconds = time.time() - t0
        try:
            cdp.send("Page.stopScreencast")
            page.wait_for_timeout(100)  # let in-flight frames arrive before closing
        except Exception:
            pass
        browser.close()
    ffmpeg.stdin.close()
    code = ffmpeg.wait()
    print(json.dumps({"status": "done" if code == 0 else "ffmpeg_failed", "out": str(out),
                      "seconds": round(seconds, 2), "frames": written[0], "screencast_frames": latest["count"],
                      "bytes": out.stat().st_size if out.exists() else 0, "size": size}), flush=True)


if __name__ == "__main__":
    main()
