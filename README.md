# chessard

A UCI chess engine that plays like a human of a chosen rating. Instead of searching for the best
move, it plays the move a person of that strength would most likely play, as predicted by the
**chessard** network.

## Quick start

Works on **Windows, macOS and Linux**. You need [Git](https://git-scm.com/downloads),
[Python](https://www.python.org/downloads/) 3.10 or newer, and about 1 GB of disk space.

**Windows** (PowerShell or Command Prompt; when installing Python, tick "Add python.exe to PATH"):
```bat
git clone https://github.com/daniel-monroe/chessard
cd chessard
py install.py
venv\Scripts\chessard.exe
```

**macOS / Linux:**
```bash
git clone https://github.com/daniel-monroe/chessard
cd chessard
python3 install.py
venv/bin/chessard
```

`install.py` sets everything up inside the `chessard` folder, then runs `verify.py` to check
that the engine works. It takes a minute or two. Re-running it only redoes what is missing.

| Step | What it does | Ends up in |
|---|---|---|
| Python | a virtualenv with the pinned versions in `requirements.txt`, plus PyTorch 2.14.1 (2.2.2 on Intel Macs, the last release for them) | `venv/` |
| Engine command | `chessard` (`chessard.exe` on Windows), a normal program to give to any chess GUI | `venv/bin/` or `venv\Scripts\` |
| Stockfish | the official Stockfish 18 release for your OS and CPU. On Linux, if no release runs (an old glibc, or ARM), it is built from source, which needs `make` and `g++` | `stockfish/` |
| Weights | downloaded from [Hugging Face](https://huggingface.co/danielgmonroe/chessard) (public, ~400 MB, no login) and checked against `weights.sha256` | `weights/` |

Options:
- `--cpu` / `--cuda`: choose the PyTorch build on Windows and Linux. The default is CUDA when an
  NVIDIA GPU is found (`nvidia-smi` works); it's a ~2.5 GB download, against ~200 MB for CPU.
  Macs always get the standard macOS build.
- `--weights-src DIR`: copy the weights from a local folder instead of downloading them.
- `--skip-verify`: don't run `verify.py` at the end.

`verify.py` checks that:
- the UCI handshake lists all five players;
- at Elo 2200 after 1.e4 d5, the first info line is `score cp 91 ... pv e4d5 string p=93.81%`
  and the move is `bestmove e4d5`;
- `--player kaufman` loads at Elo 2188;
- `--elo 1500` is rejected.

Every push is tested this way on Windows, macOS and Linux (`.github/workflows/test.yml`).

### Running it

```bash
venv/bin/chessard                          # Elo 2200   (Windows: venv\Scripts\chessard.exe)
venv/bin/chessard --elo 2400
venv/bin/chessard --player carlsen         # Carlsen's style, at his rating (2840)
```

The engine finds `weights/` and `stockfish/` in the repo on its own. To use other copies, pass
`--weights-dir` / `--stockfish` or set `CHESSARD_DIR` / `STOCKFISH`. The weights folder can be
shared between programs. The `Player` option lists every `<name>.pt` in its `loras/` subfolder,
so an adapter you train yourself shows up once you copy it in; add it to `loras/players.json` to
give it a default rating.

### GUIs and lichess-bot

Give the GUI the full path to the engine command: `C:\...\chessard\venv\Scripts\chessard.exe`
on Windows, `/.../chessard/venv/bin/chessard` on macOS and Linux. `install.py` prints it at the end.

- **Cute Chess, Arena, BanksiaGUI, En Croissant, Nibbler:** add a UCI engine with that path, then
  set `Elo`, `Player`, `Sampling` and so on in the engine options dialog.
- **cutechess-cli / fastchess:**
  ```bash
  cutechess-cli -engine cmd=/path/to/chessard/venv/bin/chessard name=chessard-2400 option.Elo=2400 \
                -engine cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo=2400 \
                -each proto=uci tc=60+1 -games 2
  ```
- **lichess-bot** (`config.yml`):
  ```yaml
  engine:
    dir: "/path/to/chessard/venv/bin/"     # Windows: C:\path\to\chessard\venv\Scripts\
    name: "chessard"                        # Windows: chessard.exe
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

**`python`/`py` is not found (Windows).** Reinstall Python from python.org and tick "Add python.exe
to PATH", or run it through the `py` launcher, which the installer adds.

**`could not create a virtualenv` (Debian/Ubuntu).** Run `sudo apt install python3-venv`.

**The weights download fails.** The repo is public, so this is usually the network (a proxy, a
firewall, or an interrupted transfer). Re-run `install.py` and it resumes. Behind a proxy, set
`HTTPS_PROXY`. You can also fetch the files another way and pass `--weights-src DIR`.

**`the weights don't match weights.sha256`.** A download was corrupted. Delete `weights/` and
re-run.

**No Stockfish release runs, and the build needs `make`.** This happens on Linux with an old
glibc, or on ARM. Install a compiler (`sudo apt install build-essential`) and re-run.

**Intel Mac: "PyTorch only goes up to Python 3.12".** PyTorch's last Intel Mac release (2.2.2)
supports Python 3.10–3.12, so run the installer with one of those (e.g. `python3.12 install.py`).

**The GUI shows no moves, or `chessard failed to load`.** Run the engine command in a terminal
and type `uci`, then `isready`; errors go to stderr. The usual causes are a wrong
`WeightsDir`/`CHESSARD_DIR` or `StockfishPath`.

**`--elo 1500` is rejected.** This is intentional, because the model was trained only on games by
2000+ players. Through `setoption`, out-of-range values are clamped to 2000-2900.

**It runs on CPU even though there's an NVIDIA GPU.** You have the CPU build of PyTorch. Re-run
`install.py --cuda` to swap it.

**`verify.py` fails only on the probability.** It allows ±0.05 points. A bigger difference means
a different Stockfish version, `SF_Depth` or weights.

**Starting over.** Delete `venv/`, `weights/` and `stockfish/` and run `install.py` again.

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
install.py, verify.py  set up from scratch on any OS / check an install
uci.py                 UCI protocol, time handling, options
chessard/inference.py  weights + LoRA loading, per-move Stockfish evals, predict()
chessard/model.py      the network
chessard/encoder.py    game -> Leela 112 input planes
```

## License

GPL-3.0. See `LICENSE`.
