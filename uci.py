#!/usr/bin/env python3
"""chessard UCI engine: plays the move a human of a given rating would most likely play.

    hf download danielgmonroe/chessard --local-dir ~/chessard-weights   # once
    ./uci.py --weights-dir ~/chessard-weights --player carlsen --elo 2840

The weights folder holds chessard.pt plus loras/<lastname>.pt, one LoRA adapter per player; any
other adapter dropped into loras/ shows up as a Player choice too. Every flag can also be set from
the GUI with `setoption`, or through $CHESSARD_DIR / $STOCKFISH, so a GUI that cannot pass
arguments can still run `./uci.py` bare.

chessard is not a searcher: its cost per move is fixed (one network forward plus a shallow
Stockfish search of every legal move), so the clock is only used as a safety cap. If `stop`
arrives or the cap runs out before the per-move searches finish, the move is chosen from a
depth-1 evaluation instead, which takes milliseconds.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import traceback

import chess

NAME = "chessard"
HF_REPO = "danielgmonroe/chessard"
BASE = "chessard.pt"
LORAS = "loras"
DEFAULT_ELO = 2200
MIN_ELO, MAX_ELO = 2000, 2900   # the model was trained on 2000+ games; below that it extrapolates
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
AUTHOR = "Daniel Monroe"
MATE_CP = 100000

# name -> (UCI type, default, min, max). Option names are matched case-insensitively.
OPTIONS = {
    "WeightsDir": ("string", os.environ.get("CHESSARD_DIR", DEFAULT_DIR), None, None),
    "StockfishPath": ("string", os.environ.get("STOCKFISH", "stockfish"), None, None),
    "Player": ("combo", "none", None, None),       # vars: the adapters found in WeightsDir
    "Elo": ("spin", 0, 0, MAX_ELO),               # 0 = the player's rating, else 2200
    "Temperature": ("spin", 100, 1, 500),          # percent; 100 = the model's own distribution
    "Sampling": ("check", False, None, None),      # sample from the distribution vs. argmax
    "SF_Depth": ("spin", 9, 1, 20),                # 9 is what the model was trained with
    "Threads": ("spin", min(8, os.cpu_count() or 1), 1, 256),  # parallel Stockfish processes
    "MultiPV": ("spin", 5, 1, 50),
    "Move Overhead": ("spin", 100, 0, 10000),
    "Ponder": ("check", False, None, None),
}
RELOAD = {"WeightsDir", "StockfishPath", "Player", "Threads"}  # changing these rebuilds the net


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def players(weights_dir: str) -> list[str]:
    """LoRA adapters available in the weights folder: every loras/<name>.pt."""
    try:
        files = os.listdir(os.path.join(os.path.expanduser(weights_dir), LORAS))
    except OSError:
        return []
    return sorted(f[:-3] for f in files if f.endswith(".pt"))


def player_info(weights_dir: str, player: str) -> dict:
    """{"name", "elo"} for a player from loras/players.json, or {} if it is not listed."""
    try:
        with open(os.path.join(os.path.expanduser(weights_dir), LORAS, "players.json")) as f:
            return json.load(f).get(player, {})
    except (OSError, ValueError):
        return {}


def uci_score(cp: int) -> str:
    """Mover-POV centipawns after a move -> UCI score from the root. Mates are stored by the
    inference code as +/-(MATE_CP - moves-to-mate-after-this-move)."""
    if cp > MATE_CP - 1000:
        return f"mate {MATE_CP - cp + 1}"
    if cp < -MATE_CP + 1000:
        return f"mate -{cp + MATE_CP}"
    return f"cp {cp}"


class Engine:
    def __init__(self) -> None:
        self.opts = {k: v[1] for k, v in OPTIONS.items()}
        self.start_fen = chess.STARTING_FEN
        self.moves: list[str] = []
        self.net = None
        self.out_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.abort = threading.Event()      # stop the computation early (stop / time cap)
        self.release = threading.Event()    # allow bestmove after `go infinite` / `go ponder`
        self.timer: threading.Timer | None = None
        self.pending_budget: float | None = None

    def elo(self) -> int:
        """The rating to imitate: Elo if set, else the selected player's own rating."""
        if self.opts["Elo"]:
            return min(MAX_ELO, max(MIN_ELO, self.opts["Elo"]))
        return player_info(self.opts["WeightsDir"], self.opts["Player"]).get("elo", DEFAULT_ELO)

    def out(self, line: str) -> None:
        with self.out_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    # ------------------------------------------------------------------ model
    def ensure_net(self):
        if self.net is None:
            from chessard import Chessard     # imports torch: deferred so `uci` answers at once
            d = os.path.expanduser(self.opts["WeightsDir"])
            weights = os.path.join(d, BASE)
            if not os.path.isfile(weights):
                raise RuntimeError(f"{weights} not found; download the weights with "
                                   f"`hf download {HF_REPO} --local-dir {d}`")
            player = self.opts["Player"]
            adapter = None if player == "none" else os.path.join(d, LORAS, f"{player}.pt")
            if adapter and not os.path.isfile(adapter):
                raise RuntimeError(f"no adapter for Player {player!r}: {adapter} not found")
            self.net = Chessard(weights, stockfish=self.opts["StockfishPath"], adapter=adapter,
                                device=os.environ.get("CHESSARD_DEVICE"),
                                sf_processes=self.opts["Threads"])
            log(f"chessard: loaded {weights} (player: {player}, Elo {self.elo()}) "
                f"on {self.net.device}")
        return self.net

    def drop_net(self) -> None:
        if self.net is not None:
            self.net.close()
            self.net = None

    # ------------------------------------------------------------------ commands
    def cmd_uci(self) -> None:
        self.out(f"id name {NAME}")
        self.out(f"id author {AUTHOR}")
        for name, (typ, default, lo, hi) in OPTIONS.items():
            if typ == "spin":
                self.out(f"option name {name} type spin default {default} min {lo} max {hi}")
            elif typ == "check":
                self.out(f"option name {name} type check default {str(default).lower()}")
            elif typ == "combo":
                choices = ["none"] + players(self.opts["WeightsDir"])
                self.out(f"option name {name} type combo default {self.opts[name]} "
                         + " ".join(f"var {c}" for c in choices))
            else:
                self.out(f"option name {name} type string default {default or '<empty>'}")
        self.out("uciok")

    def cmd_setoption(self, args: list[str]) -> None:
        if not args or args[0] != "name":
            return
        j = args.index("value") if "value" in args else len(args)
        key = " ".join(args[1:j]).lower()
        value = " ".join(args[j + 1:])
        name = next((n for n in OPTIONS if n.lower() == key), None)
        if name is None:
            log(f"chessard: unknown option {key!r}")
            return
        typ, _, lo, hi = OPTIONS[name]
        if typ == "spin":
            try:
                new = min(hi, max(lo, int(value)))
            except ValueError:
                return
        elif typ == "check":
            new = value.strip().lower() == "true"
        elif typ == "combo":
            new = value.strip().lower() or "none"
            if new != "none" and new not in players(self.opts["WeightsDir"]):
                log(f"chessard: no {LORAS}/{new}.pt in {self.opts['WeightsDir']}; keeping {self.opts[name]}")
                return
        else:
            new = "" if value.strip() in ("", "<empty>") else value.strip()
        if name in RELOAD and new != self.opts[name]:
            self.drop_net()
        self.opts[name] = new

    def cmd_position(self, args: list[str]) -> None:
        j = args.index("moves") if "moves" in args else len(args)
        if args and args[0] == "startpos":
            fen = chess.STARTING_FEN
        elif args and args[0] == "fen":
            fen = " ".join(args[1:j])
        else:
            return
        moves = args[j + 1:]
        board = chess.Board(fen)            # validate before accepting
        for u in moves:
            board.push_uci(u)
        self.start_fen, self.moves = fen, moves

    def budget(self, p: dict, white: bool) -> float | None:
        """Seconds we may spend before falling back, or None for no limit."""
        overhead = self.opts["Move Overhead"] / 1000
        if "movetime" in p:
            return max(0.01, p["movetime"] / 1000 - overhead)
        left = p.get("wtime" if white else "btime")
        if left is None:
            return None
        inc = p.get("winc" if white else "binc", 0)
        # chessard gains nothing from extra time, so this only guards against flagging.
        cap = min(left * 0.25 + inc * 0.75, left - 2 * self.opts["Move Overhead"])
        return max(0.01, cap / 1000 - overhead)

    def cmd_go(self, args: list[str]) -> None:
        self.wait()
        p, searchmoves, i = {}, [], 0
        while i < len(args):
            tok = args[i]
            if tok in ("infinite", "ponder"):
                p[tok] = True
            elif tok == "searchmoves":
                i += 1
                while i < len(args) and len(args[i]) in (4, 5) and args[i][1].isdigit():
                    searchmoves.append(args[i])
                    i += 1
                continue
            elif i + 1 < len(args):
                try:
                    p[tok] = int(args[i + 1])
                    i += 1
                except ValueError:
                    pass
            i += 1

        board = chess.Board(self.start_fen)
        for u in self.moves:
            board.push_uci(u)
        if self.timer is not None:          # e.g. armed by a ponderhit after the search finished
            self.timer.cancel()
            self.timer = None
        self.abort.clear()
        self.release.clear()
        hold = p.get("infinite") or p.get("ponder")
        if not hold:
            self.release.set()
        budget = self.budget(p, board.turn == chess.WHITE)
        self.pending_budget = budget if p.get("ponder") else None
        if budget is not None and not hold:
            self.arm_timer(budget)
        self.thread = threading.Thread(
            target=self.search, args=(board, list(self.moves), searchmoves), daemon=True)
        self.thread.start()

    def arm_timer(self, seconds: float) -> None:
        self.timer = threading.Timer(seconds, self.abort.set)
        self.timer.daemon = True
        self.timer.start()

    def cmd_ponderhit(self) -> None:
        if self.pending_budget is not None:
            self.arm_timer(self.pending_budget)
            self.pending_budget = None
        self.release.set()

    def cmd_stop(self) -> None:
        self.abort.set()
        self.release.set()

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    # ------------------------------------------------------------------ search
    def search(self, board: chess.Board, moves: list[str], searchmoves: list[str]) -> None:
        best = "0000"
        try:
            t0 = time.monotonic()
            net = self.ensure_net()
            o = self.opts
            depth = o["SF_Depth"]
            elo = self.elo()
            dist = net.predict(self.start_fen, moves, elo, o["Temperature"] / 100, depth,
                               abort=self.abort)
            if dist is None:                # stopped or out of time: decide on a cheap eval
                depth = 1
                dist = net.predict(self.start_fen, moves, elo, o["Temperature"] / 100, depth)
            if searchmoves:
                dist = [m for m in dist if m["uci"] in searchmoves] or dist
            ms = int((time.monotonic() - t0) * 1000)
            for n, m in enumerate(dist[:o["MultiPV"]], 1):
                # The ranking is by human probability, reported after `string` (which must come
                # last: it swallows the rest of the line). The score is Stockfish's eval.
                self.out(f"info depth {depth} multipv {n} score {uci_score(m['cp'])} time {ms} "
                         f"pv {m['uci']} string p={m['prob'] * 100:.2f}%")
            if dist:
                pick = (random.choices(dist, weights=[m["prob"] for m in dist])[0]
                        if o["Sampling"] else dist[0])
                best = pick["uci"]
        except Exception as e:
            log(traceback.format_exc().rstrip())
            self.out(f"info string chessard error: {e}")
            legal =[m.uci() for m in board.legal_moves if not searchmoves or m.uci() in searchmoves]
            best = legal[0] if legal else "0000"
        finally:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            self.release.wait()             # infinite/ponder: hold bestmove until stop/ponderhit
            self.out(f"bestmove {best}")

    # ------------------------------------------------------------------ loop
    def run(self) -> int:
        for raw in sys.stdin:
            parts = raw.split()
            if not parts:
                continue
            cmd, args = parts[0], parts[1:]
            try:
                if cmd == "uci":
                    self.cmd_uci()
                elif cmd == "isready":
                    try:
                        if self.thread is None:
                            self.ensure_net()
                    except Exception as e:  # still answer, or the GUI waits forever
                        log(traceback.format_exc().rstrip())
                        self.out(f"info string chessard failed to load: {e}")
                    self.out("readyok")
                elif cmd == "setoption":
                    self.wait()
                    self.cmd_setoption(args)
                elif cmd == "ucinewgame":
                    self.wait()
                    self.start_fen, self.moves = chess.STARTING_FEN, []
                elif cmd == "position":
                    self.wait()
                    self.cmd_position(args)
                elif cmd == "go":
                    self.cmd_go(args)
                elif cmd == "stop":
                    self.cmd_stop()
                elif cmd == "ponderhit":
                    self.cmd_ponderhit()
                elif cmd == "quit":
                    break
            except Exception:  # a UCI engine must never die mid-game
                log(traceback.format_exc().rstrip())
        self.cmd_stop()
        self.wait()
        self.drop_net()
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights-dir", help=f"folder with {BASE} and the player adapters "
                    "(default: $CHESSARD_DIR, else weights/ next to this script)")
    ap.add_argument("--stockfish", help="Stockfish binary (default: $STOCKFISH or 'stockfish' on PATH)")
    ap.add_argument("--player", help="play in this player's style: an adapter loras/<name>.pt in the folder")
    ap.add_argument("--elo", type=int,
                    help=f"rating to imitate, {MIN_ELO}-{MAX_ELO} "
                         f"(default: the player's own rating, else {DEFAULT_ELO})")
    ap.add_argument("--threads", type=int, help="parallel Stockfish processes")
    ap.add_argument("--sampling", action="store_true",
                    help="sample from the distribution instead of playing the most likely move")
    a = ap.parse_args()
    if a.elo is not None and a.elo != 0 and not MIN_ELO <= a.elo <= MAX_ELO:
        ap.error(f"--elo must be {MIN_ELO}-{MAX_ELO} (the model was trained on {MIN_ELO}+ games)")

    eng = Engine()
    for flag, opt in (("weights_dir", "WeightsDir"), ("stockfish", "StockfishPath"),
                      ("player", "Player"), ("elo", "Elo"), ("threads", "Threads")):
        if getattr(a, flag) is not None:
            eng.opts[opt] = getattr(a, flag)
    if a.sampling:
        eng.opts["Sampling"] = True
    eng.opts["Player"] = eng.opts["Player"].lower()
    d = os.path.expanduser(eng.opts["WeightsDir"])
    if a.weights_dir and not os.path.isfile(os.path.join(d, BASE)):
        log(f"chessard: no {BASE} in {d}; download it with `hf download {HF_REPO} --local-dir {d}`")
        return 2
    if a.player and eng.opts["Player"] != "none" and eng.opts["Player"] not in players(d):
        log(f"chessard: no {LORAS}/{eng.opts['Player']}.pt in {d} (have: {', '.join(players(d)) or 'none'})")
        return 2
    return eng.run()


if __name__ == "__main__":
    raise SystemExit(main())
