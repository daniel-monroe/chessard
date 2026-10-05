"""chessard network: Leela BT4 transformer body + a Stockfish-aware per-move policy head.

Inference only. Module names match the training checkpoint's state_dict, so weights (and LoRA
adapters, which target bt4.encoders.<i>.{wq,wk,wv,dense,ffn.dense1,ffn.dense2}) load as-is.

  - body: 1024 wide, 15 post-LN encoder layers with DeepNorm residuals and smolgen attention
    bias, 32 heads. Playing strength enters as a 32-dim vector concatenated to every input token,
    linearly interpolated between two learned anchors by Elo.
  - policy: for each legal move, MLP([trunk[from], trunk[to], this move's Stockfish expected
    score, the position's best Stockfish expected score]) -> one logit.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB = 1024
LAYERS = 15
HEADS = 32
HEAD_DIM = EMB // HEADS
DFF = 1536
INPUT_PLANES = 112
DENSE_SZ = 512                      # per-square preprocess output
SMOL_CH, SMOL_HIDDEN, SMOL_GEN = 32, 256, 256
STRENGTH_DIM = 32
ALPHA = (2.0 * LAYERS) ** -0.25     # DeepNorm residual scale
LN_EPS = 1e-3                       # Keras LayerNormalization default (weights come from lc0/TF)
VOCAB = 4352                        # 64*64 from-to moves + 256 promotions


def _ln(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=LN_EPS)


def move_to_squares() -> torch.Tensor:
    """(VOCAB, 2) move index -> (from_sq, to_sq). Promotions are rank 7 -> rank 8 only, since the
    board is mirrored when black is to move; their order is from-file, to-file, piece (q,r,b,n)."""
    t = torch.empty(VOCAB, 2, dtype=torch.long)
    i = torch.arange(4096)
    t[:4096, 0], t[:4096, 1] = i // 64, i % 64
    j = torch.arange(256)
    t[4096:, 0], t[4096:, 1] = 48 + j // 32, 56 + (j % 32) // 4
    return t


class FFN(nn.Module):
    def __init__(self, d: int, dff: int):
        super().__init__()
        self.dense1 = nn.Linear(d, dff)
        self.dense2 = nn.Linear(dff, d)

    def forward(self, x):
        return self.dense2(F.mish(self.dense1(x)))


class Smolgen(nn.Module):
    """Per-layer attention bias; the final 256 -> 64*64 projection is shared by all layers."""
    def __init__(self, global_gen: nn.Linear):
        super().__init__()
        self.compress = nn.Linear(EMB, SMOL_CH, bias=False)
        self.dense1 = nn.Linear(64 * SMOL_CH, SMOL_HIDDEN)
        self.ln1 = _ln(SMOL_HIDDEN)
        self.dense2 = nn.Linear(SMOL_HIDDEN, HEADS * SMOL_GEN)
        self.ln2 = _ln(HEADS * SMOL_GEN)
        self.global_gen = global_gen

    def forward(self, x):
        B = x.size(0)
        h = self.ln1(F.silu(self.dense1(self.compress(x).reshape(B, 64 * SMOL_CH))))
        g = self.ln2(F.silu(self.dense2(h))).reshape(B, HEADS, SMOL_GEN)
        return self.global_gen(g).reshape(B, HEADS, 64, 64)


class EncoderLayer(nn.Module):
    def __init__(self, global_gen: nn.Linear):
        super().__init__()
        self.wq = nn.Linear(EMB, EMB)
        self.wk = nn.Linear(EMB, EMB)
        self.wv = nn.Linear(EMB, EMB)
        self.dense = nn.Linear(EMB, EMB)
        self.smolgen = Smolgen(global_gen)
        self.ln1 = _ln(EMB)
        self.ffn = FFN(EMB, DFF)
        self.ln2 = _ln(EMB)

    def forward(self, x):
        B = x.size(0)
        q, k, v = (w(x).view(B, 64, HEADS, HEAD_DIM).transpose(1, 2)
                   for w in (self.wq, self.wk, self.wv))
        scores = q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM) + self.smolgen(x)
        out = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, 64, EMB)
        x = self.ln1(x + self.dense(out) * ALPHA)
        return self.ln2(x + self.ffn(x) * ALPHA)


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_preproc = nn.Linear(64 * 12, 64 * DENSE_SZ)
        self.embedding = nn.Linear(INPUT_PLANES + DENSE_SZ + STRENGTH_DIM, EMB)
        self.emb_ln = _ln(EMB)
        self.mult_gate = nn.Parameter(torch.ones(64, EMB))
        self.add_gate = nn.Parameter(torch.zeros(64, EMB))
        self.emb_ffn = FFN(EMB, DFF)
        self.emb_ffn_ln = _ln(EMB)
        self.global_smolgen = nn.Linear(SMOL_GEN, 64 * 64, bias=False)
        self.encoders = nn.ModuleList(EncoderLayer(self.global_smolgen) for _ in range(LAYERS))

    def forward(self, planes, strength):
        """planes (B,112,8,8), strength (B,STRENGTH_DIM) -> per-square embeddings (B,64,EMB)."""
        B = planes.size(0)
        x = planes.float().permute(0, 2, 3, 1).reshape(B, 64, INPUT_PLANES)
        pos = self.emb_preproc(x[..., :12].reshape(B, 64 * 12)).reshape(B, 64, DENSE_SZ)
        x = torch.cat([x, pos, strength.view(B, 1, STRENGTH_DIM).expand(B, 64, STRENGTH_DIM)], -1)
        x = self.emb_ln(F.mish(self.embedding(x))) * self.mult_gate + self.add_gate
        x = self.emb_ffn_ln(x + self.emb_ffn(x) * ALPHA)
        for enc in self.encoders:
            x = enc(x)
        return x


class ChessardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bt4 = Body()
        self.elo_vec0 = nn.Parameter(torch.zeros(STRENGTH_DIM))
        self.elo_vec3000 = nn.Parameter(torch.zeros(STRENGTH_DIM))
        self.sf_pol_fc1 = nn.Linear(2 * EMB + 2, 256)
        self.sf_pol_fc2 = nn.Linear(256, 1)
        self.register_buffer("move_to_squares", move_to_squares(), persistent=False)

    def forward(self, planes, elo, sf_move, sf_best, legal):
        """planes (B,112,8,8); elo (B,); sf_move (B,VOCAB) per-move Stockfish expected score;
        sf_best (B,) best of those; legal (B,VOCAB) bool. Returns (B,VOCAB) logits, illegal=-1e4."""
        t = (elo.float() / 3000.0).view(-1, 1)
        strength = self.elo_vec0 + t * (self.elo_vec3000 - self.elo_vec0)
        x = self.bt4(planes, strength)

        bi, mi = legal.nonzero(as_tuple=True)
        sq = self.move_to_squares[mi]
        feat = torch.cat([x[bi, sq[:, 0]], x[bi, sq[:, 1]],
                          sf_move[bi, mi].unsqueeze(1), sf_best[bi].unsqueeze(1)], dim=1)
        h = self.sf_pol_fc2(F.gelu(self.sf_pol_fc1(feat))).squeeze(1)
        logits = torch.full(legal.shape, -1e4, device=x.device, dtype=h.dtype)
        logits[bi, mi] = h
        return logits
