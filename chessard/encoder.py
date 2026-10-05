"""Encode a game (start position + moves) as Leela's classic 112 input planes.

A trimmed copy of lcztools' LeelaBoard.lcz_features (almaudoh/lczero_tools), which matches lc0:
8 history steps x (12 piece planes + 1 repetition plane), then castling rights, side to move,
the raw rule-50 counter, a zero plane and a ones plane. Planes are from the side to move's view.

History matters: the network was trained on full games, so always replay the real moves rather
than encoding a bare FEN (which would zero the 7 history steps).
"""
import collections
import struct

import chess
import numpy as np

_flat = [np.full((8, 8), i, dtype=np.uint8) for i in range(256)]
_pack_q = struct.Struct(">Q").pack

_Step = collections.namedtuple(
    "_Step", "plane_bytes repetition us_ooo us_oo them_ooo them_oo side_to_move rule50")


class LeelaBoard:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self._steps: list[_Step] = []
        self._seen = collections.Counter()
        self._record()

    def push(self, move: chess.Move) -> None:
        self.board.push(move)
        self._record()

    def _record(self) -> None:
        b = self.board
        key = b._transposition_key()
        self._seen[key] += 1
        black = 0 if b.turn else 1
        c = b.castling_rights
        w_ooo, w_oo, b_ooo, b_oo = ((c >> s) & 1 for s in (chess.A1, chess.H1, chess.A8, chess.H8))
        us_ooo, us_oo, them_ooo, them_oo = ((b_ooo, b_oo, w_ooo, w_oo) if black
                                            else (w_ooo, w_oo, b_ooo, b_oo))
        planes = b"".join(_pack_q(b.pieces_mask(pt, color))
                          for color in (True, False) for pt in range(1, 7))
        self._steps.append(_Step(planes, self._seen[key] > 1, us_ooo, us_oo, them_ooo, them_oo,
                                 black, b.halfmove_clock))

    def planes(self) -> np.ndarray:
        """(112, 8, 8) uint8 input planes for the current position."""
        cur = self._steps[-1]
        out = []
        for step in self._steps[-1:-9:-1]:            # up to 8 steps, most recent first
            p = np.unpackbits(memoryview(step.plane_bytes))[::-1].reshape(12, 8, 8)[::-1]
            if cur.side_to_move:                      # black to move: swap colours, flip ranks
                p = p.reshape(2, 6, 8, 8)[::-1, :, ::-1].reshape(12, 8, 8)
            out += [p, [_flat[step.repetition]]]
        missing = 8 - min(8, len(self._steps))
        out += [[_flat[0]] * (13 * missing)] if missing else []
        out.append([_flat[cur.us_ooo], _flat[cur.us_oo], _flat[cur.them_ooo], _flat[cur.them_oo],
                    _flat[cur.side_to_move], _flat[cur.rule50], _flat[0], _flat[1]])
        return np.concatenate(out)
