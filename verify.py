#!/usr/bin/env python3
"""Check a chessard install by talking UCI to it. Works on Linux, macOS and Windows.

    python verify.py                      # the engine install.py set up in this repo
    python verify.py --engine "CMD ARGS"  # any engine command, e.g. a custom wrapper

Checks:
  1. `uci` answers with id name chessard, the Elo option and the five player adapters
  2. Elo 2200 after 1.e4 d5: first line is `score cp 91 ... pv e4d5 string p=93.81%`
     (p within +-0.05) and `bestmove e4d5`. Deterministic on CPU and GPU.
  3. `--player kaufman` loads the adapter and logs Elo 2188 on stderr, then plays a move
  4. `--elo 1500` is rejected (exit 2, message names the 2000-2900 range)

Exits 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PLAYERS = ["carlsen", "janik", "kaufman", "nakamura", "sadler"]


def default_engine() -> list[str]:
    if os.name == "nt":
        return [os.path.join(HERE, "venv", "Scripts", "chessard.exe")]
    return [os.path.join(HERE, "venv", "bin", "chessard")]


class Engine:
    """A UCI engine subprocess with non-blocking line reads (threads, so it works on Windows)."""

    def __init__(self, cmd: list[str], timeout: float):
        self.timeout = timeout
        self.out: list[str] = []
        self.err: list[str] = []
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.out, self.lines),
                         daemon=True).start()
        threading.Thread(target=self._pump, args=(self.proc.stderr, self.err, None),
                         daemon=True).start()

    @staticmethod
    def _pump(stream, store, q):
        for line in stream:
            line = line.rstrip("\r\n")
            store.append(line)
            if q is not None:
                q.put(line)
        if q is not None:
            q.put(None)  # EOF

    def send(self, line: str) -> None:
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def expect(self, pattern: str) -> str | None:
        """Read stdout until a line matches `pattern`; None on timeout or EOF."""
        rx = re.compile(pattern)
        deadline = time.monotonic() + self.timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                line = self.lines.get(timeout=min(left, 5))
            except queue.Empty:
                continue
            if line is None:
                return None
            if rx.search(line):
                return line

    def close(self) -> None:
        self.send("quit")
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


class Report:
    def __init__(self):
        self.fails = 0

    def ok(self, cond: bool, good: str, bad: str) -> bool:
        print(f"  {'PASS' if cond else 'FAIL'} {good if cond else bad}", flush=True)
        self.fails += not cond
        return cond


def show(lines: list[str]) -> None:
    for l in lines[-30:]:
        print(f"      | {l}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", help="engine command line (default: the console script from install.py)")
    ap.add_argument("--timeout", type=float, default=600,
                    help="seconds to wait for each engine reply (default 600; first CPU load is slow)")
    a = ap.parse_args()
    cmd = shlex.split(a.engine, posix=os.name != "nt") if a.engine else default_engine()
    if not a.engine and not os.path.isfile(cmd[0]):
        print(f"verify: engine {cmd[0]} not found (run install.py first)", file=sys.stderr)
        return 2

    r = Report()
    print(f"chessard verify: {' '.join(cmd)}", flush=True)
    engines: list[Engine] = []
    try:
        # 1 + 2 ------------------------------------------------------------------
        e = Engine(cmd, a.timeout)
        engines.append(e)
        e.send("uci")
        if e.expect(r"^uciok"):
            r.ok(any(l.startswith("id name chessard") for l in e.out)
                 and any(l.startswith("option name Elo type spin") for l in e.out),
                 "uci handshake (id name chessard, Elo option)", "uci handshake incomplete")
            combo = next((l for l in e.out if l.startswith("option name Player type combo")), "")
            have = re.findall(r"\bvar (\S+)", combo)
            missing = [p for p in PLAYERS if p not in have]
            r.ok(not missing, f"player adapters listed: {' '.join(have)}",
                 f"players missing: {' '.join(missing)} (got: {' '.join(have)})")
        else:
            r.ok(False, "", "no uciok")
            show(e.out), show(e.err)

        e.send("setoption name Elo value 2200")
        e.send("isready")
        if not e.expect(r"^readyok"):
            r.ok(False, "", "no readyok")
            show(e.err)
        if any("chessard failed to load" in l for l in e.out):
            r.ok(False, "", "model failed to load")
            show(e.err)
        device = next((m.group(1) for l in e.err
                       if (m := re.search(r"chessard: loaded .* on (\S+)$", l))), "?")
        e.send("position startpos moves e2e4 d7d5")
        start = len(e.out)
        t0 = time.monotonic()
        e.send("go")
        best = e.expect(r"^bestmove")
        if best:
            secs = time.monotonic() - t0
            infos = [l for l in e.out[start:] if l.startswith("info depth") or l.startswith("bestmove")]
            print(f"      device: {device}, move took {secs:.2f}s")
            for l in infos:
                print(f"      > {l}")
            first = next((l for l in infos if " multipv 1 " in l), "")
            m = re.match(r"^info depth 9 multipv 1 score cp 91 .* pv e4d5 string p=([0-9.]+)%", first)
            p = float(m.group(1)) if m else None
            r.ok(p is not None and abs(p - 93.81) <= 0.05,
                 f"Elo 2200 after 1.e4 d5: depth 9, cp 91, pv e4d5, p={p}% (expected 93.81 +-0.05)",
                 f"unexpected first line: {first!r} "
                 "(expected: info depth 9 multipv 1 score cp 91 ... pv e4d5 string p=93.81%)")
            r.ok(best == "bestmove e4d5", "bestmove e4d5", f"got {best!r}, expected 'bestmove e4d5'")
        else:
            r.ok(False, "", f"no bestmove within {a.timeout:.0f}s")
            show(e.out), show(e.err)
        e.close()

        # 3 ----------------------------------------------------------------------
        k = Engine(cmd + ["--player", "kaufman"], a.timeout)
        engines.append(k)
        k.send("uci")
        k.expect(r"^uciok")
        k.send("isready")
        ready = k.expect(r"^readyok")
        loaded = next((l for l in k.err if "chessard: loaded" in l), "")
        if not r.ok(bool(ready) and "player: kaufman, Elo 2188" in loaded,
                    f"--player kaufman: {loaded.replace('chessard: ', '', 1)}",
                    "--player kaufman did not log 'player: kaufman, Elo 2188'"):
            show(k.err)
        k.send("position startpos moves e2e4 e7e5")
        k.send("go")
        best = k.expect(r"^bestmove [a-h][1-8][a-h][1-8]")
        if not r.ok(bool(best), f"--player kaufman plays: {best}",
                    "--player kaufman produced no bestmove"):
            show(k.out), show(k.err)
        k.close()

        # 4 ----------------------------------------------------------------------
        try:
            low = subprocess.run(cmd + ["--elo", "1500"], stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=a.timeout)
            msg = next((l for l in low.stderr.splitlines() if "error:" in l), low.stderr.strip())
            ok = low.returncode == 2 and "2000-2900" in low.stderr
            if not r.ok(ok, f"--elo 1500 rejected (exit 2: {msg})",
                        f"--elo 1500 not rejected as expected (exit {low.returncode})"):
                show(low.stderr.splitlines())
        except subprocess.TimeoutExpired:
            r.ok(False, "", "--elo 1500 did not exit")
    finally:
        for eng in engines:
            if eng.proc.poll() is None:
                eng.proc.kill()

    print()
    if r.fails == 0:
        print("verify: ALL CHECKS PASSED")
        return 0
    print(f"verify: {r.fails} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
