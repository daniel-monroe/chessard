#!/usr/bin/env bash
# chessard setup: from a fresh clone of this repo to a working chessard UCI engine.
#
#   ./setup.sh [options]
#
#   --prefix DIR        install venv/weights/stockfish/bin under DIR (default: this repo)
#   --cpu               install the CPU-only torch wheel (~200 MB instead of ~2.5 GB)
#   --cuda              install the CUDA torch wheel (default when nvidia-smi is present)
#   --weights-src DIR   copy the weights from a local folder (or set CHESSARD_WEIGHTS_SRC);
#                       default: hf download danielgmonroe/chessard
#   --python PATH       Python 3.12+ interpreter to build the venv from (or set PYTHON)
#   --sf-tag TAG        Stockfish release tag (default: sf_18)
#   --build-stockfish   build Stockfish from source instead of using the release binary
#   --skip-verify       do not run verify.sh at the end
#
# Every step is idempotent: re-running checks what is already there and only redoes what is
# missing or different. Layout under the prefix (all gitignored):
#
#   venv/        Python environment     weights/      chessard.pt, loras/ (from Hugging Face)
#   stockfish/   stockfish binary       bin/chessard  the engine command for a GUI or lichess-bot
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------- pins
TORCH_VERSION="2.14.1"                 # tested: CPU wheel 2.14.1+cpu
SF_TAG="${SF_TAG:-sf_18}"
HF_REPO="danielgmonroe/chessard"
HF_REVISION="${CHESSARD_HF_REVISION:-main}"
PY_MIN="3.12"                          # the pinned numpy 2.5.3 needs 3.12+

# ----------------------------------------------------------------------- args
PREFIX="${CHESSARD_PREFIX:-$HERE}"
WEIGHTS_SRC="${CHESSARD_WEIGHTS_SRC:-}"
PY="${PYTHON:-}"
TORCH_FLAVOR=auto
BUILD_SF=0
VERIFY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift ;;
    --cpu) TORCH_FLAVOR=cpu ;;
    --cuda) TORCH_FLAVOR=cuda ;;
    --weights-src) WEIGHTS_SRC="$2"; shift ;;
    --python) PY="$2"; shift ;;
    --sf-tag) SF_TAG="$2"; shift ;;
    --build-stockfish) BUILD_SF=1 ;;
    --skip-verify) VERIFY=0 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "this script targets Linux (found $(uname -s))"
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
CODE="$HERE"; VENV="$PREFIX/venv"; WEIGHTS="$PREFIX/weights"
SFDIR="$PREFIX/stockfish"; BIN="$PREFIX/bin"
T0=$(date +%s)
say "installing chessard into $PREFIX"

# ------------------------------------------------------- 1. prerequisites
say "1/6 prerequisites"
for c in tar sha256sum; do command -v "$c" >/dev/null || die "missing required command: $c"; done
command -v curl >/dev/null || command -v wget >/dev/null || die "need curl or wget"
info "ok: tar, sha256sum, $(command -v curl >/dev/null && echo curl || echo wget)"

# ------------------------------------------------------------ 2. python
say "2/6 Python >= $PY_MIN"
new_enough() { "$1" -c "import sys; sys.exit(0 if sys.version_info >= (${PY_MIN%.*}, ${PY_MIN#*.}) else 1)" 2>/dev/null; }
has_venv() { "$1" -c 'import venv, ensurepip' 2>/dev/null; }
if [ -n "$PY" ]; then
  new_enough "$PY" || die "$PY is not Python $PY_MIN+ ($("$PY" -V 2>&1 || echo 'not runnable'))"
elif [ -x "$VENV/bin/python" ] && new_enough "$VENV/bin/python"; then
  PY="$VENV/bin/python"              # re-run: keep the interpreter the venv was built with
else
  for c in python3 python3.14 python3.13 python3.12; do
    if command -v "$c" >/dev/null 2>&1 && new_enough "$c" && has_venv "$c"; then PY="$(command -v "$c")"; break; fi
  done
  if [ -z "$PY" ] && command -v uv >/dev/null 2>&1; then
    PY="$(uv python find ">=$PY_MIN" 2>/dev/null || true)"
    if [ -z "$PY" ]; then
      info "no Python $PY_MIN+ found; installing one with uv (uv python install $PY_MIN)"
      uv python install "$PY_MIN" >/dev/null
      PY="$(uv python find ">=$PY_MIN")"
    fi
  fi
fi
[ -n "$PY" ] || die "chessard needs Python $PY_MIN or newer (found: $(python3 -V 2>&1 || echo none)).
       Install one (e.g. 'apt install python3.12 python3.12-venv', or
       'curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install $PY_MIN')
       and re-run, or pass --python /path/to/python3.12."
info "using $("$PY" -c 'import platform;print(platform.python_version())') ($PY)"

chmod +x "$CODE/uci.py"

# -------------------------------------------------- 3. venv + pinned deps
say "3/6 virtualenv + pinned Python packages -> $VENV"
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c 'import sys' 2>/dev/null; then
  info "existing venv is broken; recreating"; rm -rf "$VENV"
fi
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --disable-pip-version-check --upgrade pip

if [ "$TORCH_FLAVOR" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then TORCH_FLAVOR=cuda; else TORCH_FLAVOR=cpu; fi
fi
have_torch="$("$VPY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
torch_ok() {  # is the installed torch the pinned version in the requested flavour?
  case "$TORCH_FLAVOR:$have_torch" in
    cpu:"$TORCH_VERSION+cpu") return 0 ;;
    cuda:"$TORCH_VERSION"|cuda:"$TORCH_VERSION+cu"*) return 0 ;;
  esac
  return 1
}
if torch_ok; then
  info "torch $have_torch already installed"
else
  # pip treats 2.14.1+cpu as satisfying ==2.14.1, so switching flavour needs an uninstall.
  [ -n "$have_torch" ] && { info "replacing torch $have_torch"; "$VPY" -m pip uninstall --quiet -y torch; }
  if [ "$TORCH_FLAVOR" = cpu ]; then
    info "installing torch $TORCH_VERSION (CPU build, ~200 MB)"
    "$VPY" -m pip install --quiet --disable-pip-version-check "torch==$TORCH_VERSION+cpu" \
        --index-url https://download.pytorch.org/whl/cpu
  else
    info "installing torch $TORCH_VERSION (CUDA build from PyPI, ~2.5 GB; use --cpu to avoid)"
    "$VPY" -m pip install --quiet --disable-pip-version-check "torch==$TORCH_VERSION"
  fi
fi
info "installing requirements.lock"
"$VPY" -m pip install --quiet --disable-pip-version-check -r "$HERE/requirements.lock"
"$VPY" - <<'PY'
import chess, numpy, torch
print(f"    torch {torch.__version__} (CUDA available: {torch.cuda.is_available()}), "
      f"chess {chess.__version__}, numpy {numpy.__version__}")
PY

# ------------------------------------------------------------- 4. stockfish
say "4/6 Stockfish $SF_TAG -> $SFDIR/stockfish"
sf_args=(--dest "$SFDIR" --tag "$SF_TAG")
[ "$BUILD_SF" = 1 ] && sf_args+=(--build)
bash "$HERE/scripts/get-stockfish.sh" "${sf_args[@]}"

# --------------------------------------------------------------- 5. weights
say "5/6 weights -> $WEIGHTS"
mkdir -p "$WEIGHTS/loras"
check_weights() { ( cd "$WEIGHTS" && sha256sum --quiet -c "$HERE/weights.sha256" ) >/dev/null 2>&1; }
if check_weights; then
  info "all files present and checksums match"
elif [ -n "$WEIGHTS_SRC" ]; then
  [ -f "$WEIGHTS_SRC/chessard.pt" ] || die "no chessard.pt in --weights-src $WEIGHTS_SRC"
  info "copying from $WEIGHTS_SRC"
  ( cd "$WEIGHTS_SRC" && find . -path ./.cache -prune -o -type f -print ) | while read -r f; do
    mkdir -p "$WEIGHTS/$(dirname "$f")"
    cmp -s "$WEIGHTS_SRC/$f" "$WEIGHTS/$f" || cp -f "$WEIGHTS_SRC/$f" "$WEIGHTS/$f"
  done
else
  info "downloading $HF_REPO@$HF_REVISION from Hugging Face (~400 MB, public, no login needed)"
  "$VENV/bin/hf" download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$WEIGHTS" >/dev/null \
    || die "hf download $HF_REPO failed (network? proxy? see the output above).
       Re-run to resume, or pass --weights-src DIR with a local copy of the weights."
fi
check_weights || { ( cd "$WEIGHTS" && sha256sum -c "$HERE/weights.sha256" ) >&2 || true
  die "weights in $WEIGHTS do not match weights.sha256 (corrupt download, or the published
       weights changed: compare with the Hugging Face repo, then update weights.sha256)"; }
info "verified $(wc -l < "$HERE/weights.sha256") files against weights.sha256"

# --------------------------------------------------------------- 6. wrapper
say "6/6 wrapper -> $BIN/chessard"
mkdir -p "$BIN"
cat > "$BIN/chessard.tmp" <<EOF
#!/bin/sh
# chessard UCI engine (generated by setup.sh). All uci.py flags pass through,
# e.g.  chessard --player carlsen   |   chessard --elo 2300
: "\${CHESSARD_DIR:=$WEIGHTS}"
: "\${STOCKFISH:=$SFDIR/stockfish}"
export CHESSARD_DIR STOCKFISH
exec "$VPY" "$CODE/uci.py" "\$@"
EOF
chmod +x "$BIN/chessard.tmp"; mv "$BIN/chessard.tmp" "$BIN/chessard"
info "engine command: $BIN/chessard"

say "setup finished in $(( $(date +%s) - T0 ))s"

if [ "$VERIFY" = 1 ]; then
  bash "$HERE/verify.sh" --prefix "$PREFIX"
fi

cat <<EOF

Run the engine:          $BIN/chessard                (Elo 2200, or --elo 2000-2900)
Play as a player:        $BIN/chessard --player carlsen
Re-check at any time:    $HERE/verify.sh --prefix $PREFIX
EOF
