# chessard-uci

A UCI chess engine that plays like a human of a chosen rating. Instead of searching for the best
move, it plays the move a person of that strength would most likely play, as predicted by the
**chessard** network.

```
position startpos moves e2e4 d7d5
setoption name Elo value 2200
go
info depth 9 multipv 1 score cp 91 time 262 pv e4d5 string p=93.81%
info depth 9 multipv 2 score cp -48 time 262 pv b1c3 string p=3.32%
info depth 9 multipv 3 score cp -3 time 262 pv e4e5 string p=1.59%
bestmove e4d5
```

`score` is Stockfish's evaluation of each move; `p=` is the probability that a human of the set
rating plays it, which is what chessard ranks by.

## What you need

This repo is code only. You also need:

- **The weights** from Hugging Face: `chessard.pt` (~400 MB) plus a small LoRA adapter per
  player in `loras/` (`carlsen.pt`, `nakamura.pt`, `sadler.pt`, `janik.pt`, `kaufman.pt`).
- **[Stockfish](https://stockfishchess.org/download/)**: any recent version. The network takes a
  Stockfish evaluation of every legal move as input.
- **Python 3.10+** with `pip install -r requirements.txt`. A CUDA build of PyTorch is used
  automatically if present (~0.25 s/move on a GPU); CPU works too (~0.4 s/move on 8 cores).

## Run

```bash
pip install -r requirements.txt
hf download danielgmonroe/chessard --local-dir ~/chessard-weights
./uci.py --weights-dir ~/chessard-weights --stockfish /path/to/stockfish --elo 2300
./uci.py --weights-dir ~/chessard-weights --player carlsen              # Carlsen, at 2840
```

The weights folder can be shared: set `CHESSARD_DIR=~/chessard-weights` once and drop
`--weights-dir`. If neither is given, the engine looks in `weights/` next to `uci.py`. The
`Player` option lists every `<name>.pt` in the folder's `loras/` subfolder, so an adapter you
train yourself appears there once you copy it in; add it to `loras/players.json` to give it a
default rating.

Point any UCI GUI (Cutechess, Arena, BanksiaGUI, lichess-bot, fastchess) at `uci.py`. If a GUI
can't pass arguments, use the environment variables (`CHESSARD_DIR`, `STOCKFISH`) or set the
`WeightsDir` and `StockfishPath` options from the GUI. If it needs a plain executable, use a
wrapper:

```bash
#!/bin/sh
exec /path/to/venv/bin/python /path/to/chessard-uci/uci.py --weights-dir ~/chessard-weights "$@"
```

## Options

| Option | Default | |
|---|---|---|
| `Elo` | 0 (auto) | rating to imitate, 2000–2900: the model was trained only on games by 2000+ players, and below that its behaviour is not meaningful. 0 means the selected player's own rating from `loras/players.json`, or 2200 with no player |
| `Player` | none | play in one player's style using their adapter: carlsen (2840), nakamura (2810), sadler (2692), janik (2504), kaufman (2188) |
| `Sampling` | false | sample a move from the distribution instead of always playing the most likely one; gives varied, more human games |
| `Temperature` | 100 | percent; >100 flattens the distribution, <100 sharpens it |
| `MultiPV` | 5 | how many candidate moves to report |
| `SF_Depth` | 9 | depth of the per-move Stockfish searches. The network was trained on depth 9; other depths work but shift its predictions |
| `Threads` | min(8, cores) | Stockfish processes run in parallel |
| `Move Overhead` | 100 | ms reserved for GUI/network lag |
| `WeightsDir`, `StockfishPath` | | same as `--weights-dir`, `--stockfish` |

Set the `CHESSARD_DEVICE` environment variable (`cpu`, `cuda`, `cuda:1`) to choose a device.

## Time and `stop`

chessard does not search, so extra time doesn't make it better: each move costs one network
forward pass plus a shallow Stockfish search of every legal move. The clock only acts as a safety
cap (25% of remaining time plus most of the increment). If that runs out, or `stop` arrives,
before the searches finish, the move is chosen from depth-1 evaluations instead, which is much
cheaper. `go infinite` and `go ponder` hold `bestmove` until `stop` / `ponderhit`, as UCI requires.
`searchmoves` is supported; `depth`, `nodes` and `mate` limits are accepted and ignored.

## Python API

```python
import chess
from chessard import Chessard

net = Chessard("chessard-weights/chessard.pt", stockfish="stockfish",
               adapter="chessard-weights/loras/carlsen.pt")   # adapter is optional
for m in net.predict(chess.STARTING_FEN, ["e2e4", "e7e5"], elo=2200)[:3]:
    print(m["uci"], f"{m['prob']:.1%}", m["cp"])
net.close()
```

Pass the real move sequence rather than a bare FEN. The network sees the last 8 positions, so
predictions from a FEN with no history are worse.

## The model

A Leela Chess Zero BT4 transformer body (1024 wide, 15 layers, 32 heads) reimplemented in
PyTorch, on Leela's 112-plane input with 8 steps of history. Playing strength enters as a learned
vector interpolated by Elo, and the policy head scores each legal move from the embeddings of its
from/to squares plus the Stockfish expected score of that move and of the best move.

```
uci.py                 UCI protocol, time handling, options
chessard/inference.py  weights + LoRA loading, per-move Stockfish evals, predict()
chessard/model.py      the network
chessard/encoder.py    game -> Leela 112 input planes
```

## License

GPL-3.0. See `LICENSE`.
