#!/usr/bin/env bash
# Install a pinned Stockfish release binary to <dest>/stockfish. Tries the official prebuilt
# release first (best CPU variant, then more portable ones) and falls back to building that
# same tag from source, which is needed on hosts whose glibc is older than the release
# binaries expect (e.g. Ubuntu 20.04).
#
#   ./get-stockfish.sh --dest DIR             # prebuilt, else build from source
#   ./get-stockfish.sh --dest DIR --build     # always build from source
#   ./get-stockfish.sh --dest DIR --force     # replace an existing working binary
#   ./get-stockfish.sh --dest DIR --tag sf_17.1
#
# Idempotent: if <dest>/stockfish already runs and speaks UCI, nothing is done.
set -euo pipefail

SF_TAG="${SF_TAG:-sf_18}"
DEST=""
FORCE=0
BUILD_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift ;;
    --tag) SF_TAG="$2"; shift ;;
    --force) FORCE=1 ;;
    --build) BUILD_ONLY=1 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$DEST" ] || { echo "usage: $0 --dest DIR [--tag sf_18] [--build] [--force]" >&2; exit 2; }

say()  { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

BIN="$DEST/stockfish"
# Does a candidate binary actually start and speak UCI on this machine?
works() { [ -x "$1" ] && printf 'uci\nquit\n' | "$1" 2>/dev/null | grep -q '^uciok'; }
sf_name() { printf 'uci\nquit\n' | "$1" 2>/dev/null | sed -n 's/^id name //p' | head -1; }

if [ "$FORCE" = 0 ] && works "$BIN"; then
  say "Stockfish already installed: $(sf_name "$BIN") ($BIN)"
  exit 0
fi

fetch() {  # fetch <url> <outfile>
  if command -v curl >/dev/null; then curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null; then wget -q -O "$2" "$1"
  else die "need curl or wget"; fi
}

OS="$(uname -s)"; MACH="$(uname -m)"
FLAGS=""
[ -r /proc/cpuinfo ] && FLAGS="$(grep -m1 '^flags' /proc/cpuinfo || true)"
[ "$OS" = Darwin ] && FLAGS="$(sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null | tr 'A-Z' 'a-z' || true)"
has() { case " $FLAGS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Best micro-architecture this CPU supports, in Stockfish's ARCH naming.
if [ "$MACH" = arm64 ] || { [ "$MACH" = aarch64 ] && [ "$OS" = Darwin ]; }; then
  ARCH=apple-silicon
elif [ "$MACH" = aarch64 ]; then
  ARCH=armv8
elif has avx512f && has avx512vnni; then ARCH=x86-64-vnni512
elif has avx512f;                  then ARCH=x86-64-avx512
elif has bmi2;                     then ARCH=x86-64-bmi2
elif has avx2;                     then ARCH=x86-64-avx2
elif has sse4_1 && has popcnt;     then ARCH=x86-64-sse41-popcnt
else                                    ARCH=x86-64
fi
say "Stockfish $SF_TAG for $MACH, ARCH=$ARCH"

mkdir -p "$DEST"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

try_prebuilt() {
  case "$OS" in
    Linux)  PLAT=ubuntu ;;
    Darwin) PLAT=macos ;;
    *) warn "no prebuilt release for $OS"; return 1 ;;
  esac
  case "$ARCH" in
    apple-silicon) CANDS="m1-apple-silicon" ;;
    armv8)         warn "no prebuilt Linux arm64 release"; return 1 ;;
    *)             CANDS="$ARCH x86-64-avx2 x86-64-sse41-popcnt x86-64" ;;
  esac
  for c in $CANDS; do
    asset="stockfish-$PLAT-$c.tar"
    url="https://github.com/official-stockfish/Stockfish/releases/download/$SF_TAG/$asset"
    say "trying prebuilt $asset"
    rm -f "$TMP/sf.tar"
    fetch "$url" "$TMP/sf.tar" 2>/dev/null || { warn "not available: $asset"; continue; }
    rm -rf "$TMP/x"; mkdir -p "$TMP/x"
    tar -xf "$TMP/sf.tar" -C "$TMP/x" || { warn "bad archive: $asset"; continue; }
    cand="$(find "$TMP/x" -type f -name 'stockfish*' ! -name '*.tar' -perm -u+x | head -1)"
    [ -n "$cand" ] || cand="$(find "$TMP/x" -type f -name 'stockfish-*' | head -1)"
    [ -n "$cand" ] || { warn "no binary inside $asset"; continue; }
    chmod +x "$cand"
    if works "$cand"; then install -m 0755 "$cand" "$BIN"; return 0; fi
    warn "$asset does not run on this host (usually: glibc older than the release needs)"
  done
  return 1
}

build_from_source() {
  command -v make >/dev/null || die "building Stockfish needs make (apt install build-essential)"
  command -v g++  >/dev/null || command -v clang++ >/dev/null || die "building Stockfish needs g++ or clang++ (apt install build-essential)"
  SRC="$TMP/src"
  say "downloading Stockfish $SF_TAG source"
  fetch "https://github.com/official-stockfish/Stockfish/archive/refs/tags/$SF_TAG.tar.gz" "$TMP/src.tar.gz" \
    || die "could not download the $SF_TAG source tarball"
  mkdir -p "$SRC"
  tar -xzf "$TMP/src.tar.gz" -C "$SRC" --strip-components=1
  NPROC="$( { nproc || sysctl -n hw.ncpu; } 2>/dev/null || echo 4)"
  say "building Stockfish $SF_TAG from source (ARCH=$ARCH, -j$NPROC); takes a few minutes"
  # profile-build (PGO) is ~10% faster; the plain build is the portable fallback.
  ( cd "$SRC/src" && make -j"$NPROC" profile-build ARCH="$ARCH" >"$TMP/build.log" 2>&1 ) \
    || ( cd "$SRC/src" && make clean >/dev/null 2>&1; make -j"$NPROC" build ARCH="$ARCH" >"$TMP/build.log" 2>&1 ) \
    || { tail -30 "$TMP/build.log" >&2; die "Stockfish build failed"; }
  install -m 0755 "$SRC/src/stockfish" "$BIN"
}

if [ "$BUILD_ONLY" = 1 ] || ! try_prebuilt; then
  build_from_source
fi

works "$BIN" || die "installed binary at $BIN does not run"
say "installed $(sf_name "$BIN") -> $BIN"
