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

## Quick start

Linux, Python 3.12+, about 1 GB of disk space:

```bash
git clone https://github.com/daniel-monroe/chessard-uci && cd chessard-uci
./setup.sh            # add --cpu to skip the ~2.5 GB CUDA build of PyTorch
./verify.sh           # talks UCI to the engine and checks a known result
bin/chessard          # the engine: give this path to your GUI or lichess-bot
```

`setup.sh` installs everything inside the repo folder (all gitignored), and re-running it only
redoes what is missing:

| Step | What it does | Ends up in |
|---|---|---|
| Python | finds Python 3.12+ (or installs one with `uv`) and creates a venv with the exact versions in `requirements.lock` plus a pinned PyTorch | `venv/` |
| Stockfish | downloads the pinned official release (sf_18) for your CPU; builds it from source if the binary won't run (old glibc) | `stockfish/stockfish` |
| Weights | `hf download danielgmonroe/chessard` (public, ~400 MB, no login), then checks it against `weights.sha256` | `weights/` |
| Wrapper | a script that runs `uci.py` with the right Python, weights and Stockfish | `bin/chessard` |

Options: `--cpu` / `--cuda` to choose the PyTorch build (default: CUDA if `nvidia-smi` works),
`--prefix DIR` to install somewhere else, `--weights-src DIR` (or `CHESSARD_WEIGHTS_SRC`) to copy
the weights from a local folder instead of downloading, `--python PATH`, and
`--build-stockfish`. Run `./setup.sh --help` for the full list.

`verify.sh` checks that:
- the UCI handshake lists all five players;
- at Elo 2200 after 1.e4 d5 the first info line is `score cp 91 ... pv e4d5 string p=93.81%` and
  the move is `bestmove e4d5` (deterministic on CPU and GPU);
- `--player kaufman` loads at Elo 2188;
- `--elo 1500` is rejected.

### Running it

```bash
bin/chessard                          # Elo 2200
bin/chessard --elo 2400
bin/chessard --player carlsen         # Carlsen's style, at his rating (2840)
```

`bin/chessard` accepts all of `uci.py`'s flags. To run `uci.py` yourself instead, point it at
a weights folder with `--weights-dir` or `CHESSARD_DIR`, and at Stockfish with `--stockfish` or
`STOCKFISH`. The weights folder can be shared between programs. The `Player` option lists every
`<name>.pt` in its `loras/` subfolder, so an adapter you train yourself shows up once you copy
it in; add it to `loras/players.json` to give it a default rating.

### GUIs and lichess-bot

Give the GUI the absolute path to `bin/chessard`, since most GUIs don't expand `~`.

- **Cute Chess, Arena, BanksiaGUI, En Croissant, Nibbler:** add a UCI engine with that path, then
  set `Elo`, `Player`, `Sampling` and so on in the engine options dialog.
- **cutechess-cli / fastchess:**
  ```bash
  cutechess-cli -engine cmd=/path/to/chessard-uci/bin/chessard name=chessard-2400 option.Elo=2400 \
                -engine cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo=2400 \
                -each proto=uci tc=60+1 -games 2
  ```
- **lichess-bot** (`config.yml`):
  ```yaml
  engine:
    dir: "/path/to/chessard-uci/bin/"
    name: "chessard"
    protocol: "uci"
    uci_options:
      Elo: 2400            # 2000-2900, or 0 for the player's own rating
      Player: "none"       # or carlsen, nakamura, sadler, janik, kaufman
      Sampling: true       # vary moves between games
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

## Troubleshooting

**`hf download` fails.** The weights repo is public, so this is usually the network (a proxy,
a firewall, or an interrupted transfer). Re-run `./setup.sh` and the download resumes. Behind a
proxy, set `HTTPS_PROXY`. You can also fetch the files some other way and pass
`--weights-src DIR`.

**`weights ... do not match weights.sha256`.** A download was corrupted. Delete `weights/` and
re-run `./setup.sh`.

**`chessard needs Python 3.12 or newer`.** Install one (`apt install python3.12 python3.12-venv`,
or `uv python install 3.12`) or pass `--python /path/to/python3.12`. If `python3 -m venv` says
"ensurepip is not available", install `python3.12-venv`.

**Every Stockfish release "does not run on this host".** This is normal on distributions with
an older glibc (e.g. Ubuntu 20.04). The script then builds Stockfish from source, which needs
`make` and `g++` (`apt install build-essential`) and internet access.

**The GUI shows no moves, or `chessard failed to load`.** Run `bin/chessard` in a terminal and
type `uci`, then `isready`; errors go to stderr. The usual causes are a wrong
`WeightsDir`/`CHESSARD_DIR` (no `chessard.pt` there) or a wrong `StockfishPath`.

**`--elo 1500` is rejected.** This is intentional, because the model was trained only on games by
2000+ players. Through `setoption`, out-of-range values are clamped to 2000-2900.

**It runs on CPU even though there's a GPU.** You have the CPU build of PyTorch (its version
ends in `+cpu`). Run `./setup.sh --cuda` to swap it.

**`verify.sh` fails only on the probability.** It allows ±0.05 points. A bigger difference means
a different Stockfish version, `SF_Depth` or weights.

**Starting over.** Delete `venv/ weights/ stockfish/ bin/` and run `./setup.sh` again.

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
setup.sh, verify.sh    install from scratch / check an install (scripts/get-stockfish.sh)
uci.py                 UCI protocol, time handling, options
chessard/inference.py  weights + LoRA loading, per-move Stockfish evals, predict()
chessard/model.py      the network
chessard/encoder.py    game -> Leela 112 input planes
```

## License

GPL-3.0. See `LICENSE`.
