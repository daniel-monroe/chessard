"""Load chessard and turn a game into a probability distribution over human moves.

    net = Chessard("chessard.pt", stockfish="stockfish")
    moves = net.predict(chess.STARTING_FEN, ["e2e4", "e7e5"], elo=1400)
    # [{"uci": "g1f3", "prob": 0.86, "cp": 41}, ...] sorted by prob

The policy head needs a Stockfish evaluation of every legal move, computed exactly as in
training: push the move, search the child at a fixed depth (9 by default) with a cleared hash,
and convert the mover's centipawns to an expected score 1/(1+10^(-cp/400)). A move that mates
scores 1.0 and any other game-ending move (cp 0) 0.5. These searches dominate the cost, so they run in
parallel across a pool of single-threaded Stockfish processes (single-threaded keeps them
deterministic).
"""
from __future__ import annotations

import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import LeelaBoard
from .model import VOCAB, ChessardModel

MATE_CP = 100000


def _vocab() -> dict[str, int]:
    """uci -> model move index. Black's moves are mirrored to white's view first."""
    names = [chess.square_name(a) + chess.square_name(b) for a in range(64) for b in range(64)]
    names += [f"{f}7{t}8{p}" for f in "abcdefgh" for t in "abcdefgh" for p in "qrbn"]
    return {m: i for i, m in enumerate(names)}


MOVE_INDEX = _vocab()


def _mirror(uci: str) -> str:
    return uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]


def _expected_score(cp: int) -> float:
    return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))


class _LoRALinear(nn.Linear):
    """nn.Linear plus a low-rank update: y = Wx + b + (alpha/r) * B(Ax)."""

    @classmethod
    def wrap(cls, lin: nn.Linear, rank: int, alpha: float) -> "_LoRALinear":
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                device=lin.weight.device)
        m.weight, m.bias = lin.weight, lin.bias
        m.lora_A = nn.Parameter(torch.zeros(rank, lin.in_features, device=lin.weight.device))
        m.lora_B = nn.Parameter(torch.zeros(lin.out_features, rank, device=lin.weight.device))
        m.lora_scale = float(alpha) / float(rank)
        return m

    def forward(self, x):
        return super().forward(x) + self.lora_scale * F.linear(F.linear(x, self.lora_A), self.lora_B)


def _apply_adapter(model: nn.Module, path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["adapter_state_dict"]
    for target in sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith((".lora_A", ".lora_B"))}):
        parent, _, name = target.rpartition(".")
        parent = model.get_submodule(parent)
        setattr(parent, name, _LoRALinear.wrap(getattr(parent, name), ck["lora_rank"], ck["lora_alpha"]))
    model.load_state_dict(sd, strict=False)


def load_model(weights: str, adapter: str | None = None, device: str = "cpu") -> ChessardModel:
    model = ChessardModel()
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    sd = ck.get("model_state_dict", ck)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, _ = model.load_state_dict(sd, strict=False)  # the unused value head is "unexpected"
    if missing:
        raise RuntimeError(f"{weights} is not a chessard checkpoint (missing {missing[:3]}...)")
    if adapter:
        _apply_adapter(model, adapter)
    return model.to(device).eval()


class Chessard:
    def __init__(self, weights: str, stockfish: str = "stockfish", adapter: str | None = None,
                 device: str | None = None, sf_processes: int = 4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(weights, adapter, self.device)
        self._engines: list[chess.engine.SimpleEngine] = []
        self._free: queue.Queue = queue.Queue()
        try:
            for _ in range(max(1, sf_processes)):
                e = chess.engine.SimpleEngine.popen_uci(stockfish, stderr=sys.stderr)
                e.configure({"Threads": 1, "Hash": 128})
                self._engines.append(e)
                self._free.put(e)
        except BaseException:
            self.close()
            raise
        self._pool = ThreadPoolExecutor(max_workers=len(self._engines))

    def close(self) -> None:
        if hasattr(self, "_pool"):
            self._pool.shutdown(wait=True)
        for e in self._engines:
            try:
                e.quit()
            except Exception:
                pass
        self._engines = []

    def _child_eval(self, child: chess.Board, mover: chess.Color, depth: int, abort):
        if abort is not None and abort.is_set():
            return None
        eng = self._free.get()
        try:
            # game=object() makes python-chess send ucinewgame, clearing the hash so every child
            # is scored independently of search order, as in the training annotations.
            info = eng.analyse(child, chess.engine.Limit(depth=depth), game=object())
        finally:
            self._free.put(eng)
        return info["score"].pov(mover).score(mate_score=MATE_CP)

    def _stockfish_scores(self, board: chess.Board, depth: int, abort) -> dict[str, int] | None:
        """{uci: mover-POV centipawns after the move}, or None if `abort` was set."""
        cps: dict[str, int] = {}
        jobs = {}
        for move in board.legal_moves:
            child = board.copy()
            child.push(move)
            if child.is_checkmate():
                cps[move.uci()] = MATE_CP
            elif child.is_game_over():
                cps[move.uci()] = 0
            else:
                jobs[move.uci()] = self._pool.submit(self._child_eval, child, board.turn, depth, abort)
        for uci, fut in jobs.items():
            cp = fut.result()
            if cp is None:
                return None
            cps[uci] = int(cp)
        return cps

    @torch.no_grad()
    def predict(self, start_fen: str, moves: list[str], elo: int, temperature: float = 1.0,
                depth: int = 9, abort: threading.Event | None = None) -> list[dict] | None:
        """Human-move distribution for the position after `moves` from `start_fen`, sorted by
        probability: [{"uci", "prob", "cp"}]. Empty if the game is over; None if aborted."""
        lb = LeelaBoard(start_fen)
        for u in moves:
            lb.push(chess.Move.from_uci(u))
        board = lb.board
        if board.is_game_over(claim_draw=True):
            return []

        index: dict[int, str] = {}
        for m in board.legal_moves:
            index[MOVE_INDEX[m.uci() if board.turn == chess.WHITE else _mirror(m.uci())]] = m.uci()
        cps = self._stockfish_scores(board, depth, abort)
        if cps is None:
            return None

        legal = torch.zeros(1, VOCAB, dtype=torch.bool)
        sf_move = torch.full((1, VOCAB), 0.5)
        for i, uci in index.items():
            legal[0, i] = True
            sf_move[0, i] = _expected_score(cps[uci])   # mate -> 1.0, draw (0) -> 0.5
        idx = sorted(index)
        sf_best = sf_move[0, idx].max().view(1)

        dev = self.device
        planes = torch.from_numpy(lb.planes()).unsqueeze(0).float()
        logits = self.model(planes.to(dev), torch.tensor([elo], device=dev), sf_move.to(dev),
                            sf_best.to(dev), legal.to(dev))[0].float()[idx]
        probs = torch.softmax(logits / max(temperature, 1e-3), dim=0).tolist()
        out = [{"uci": index[i], "prob": p, "cp": cps[index[i]]} for i, p in zip(idx, probs)]
        return sorted(out, key=lambda m: m["prob"], reverse=True)
