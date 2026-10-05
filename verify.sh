#!/usr/bin/env bash
# Check a chessard install by talking UCI to it. Exits non-zero if any check fails.
#
#   ./verify.sh                       # the install made by ./setup.sh in this repo
#   ./verify.sh --prefix DIR          # the --prefix given to setup.sh, if you used one
#   ./verify.sh --engine CMD          # any engine command, e.g. a custom wrapper
#
# Checks:
#   1. `uci` answers with id name chessard, the Elo option and the five player adapters
#   2. Elo 2200 after 1.e4 d5: first line is `score cp 91 ... pv e4d5 string p=93.81%`
#      (p within +-0.05) and `bestmove e4d5`. Deterministic, on CPU and GPU.
#   3. `--player kaufman` loads the adapter and logs Elo 2188 on stderr, then plays a move
#   4. `--elo 1500` is rejected (exit 2, message names the 2000-2900 range)
set -uo pipefail
export LC_ALL=C

PREFIX="${CHESSARD_PREFIX:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ENGINE=""
TIMEOUT="${VERIFY_TIMEOUT:-600}"     # seconds per wait; the first model load on CPU is the slow part
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift ;;
    --engine) ENGINE="$2"; shift ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$ENGINE" ] || ENGINE="$PREFIX/bin/chessard"
[ -x "${ENGINE%% *}" ] || { echo "verify: engine $ENGINE not found or not executable (run setup.sh first)" >&2; exit 2; }

LOG="$(mktemp -d)"; trap 'kill "${ENG_PID:-}" 2>/dev/null; rm -rf "$LOG"' EXIT
FAILS=0
pass() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

# --- tiny UCI driver: start the engine as a coprocess, send lines, wait for a pattern
start() {  # start <name> <engine args...>
  NAME="$1"; shift
  : > "$LOG/$NAME.out"
  # shellcheck disable=SC2086
  coproc ENG { exec $ENGINE "$@" 2>"$LOG/$NAME.err"; }
  ENG_PID=$ENG_PID
}
send() { printf '%s\n' "$*" >&"${ENG[1]}"; }
expect() {  # expect <regex>: read engine output until a line matches; 1 on timeout/EOF
  local line deadline=$(( $(date +%s) + TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    IFS= read -r -t 5 line <&"${ENG[0]}" || { kill -0 "$ENG_PID" 2>/dev/null && continue; return 1; }
    printf '%s\n' "$line" >> "$LOG/$NAME.out"
    [[ "$line" =~ $1 ]] && { MATCH="$line"; return 0; }
  done
  return 1
}
finish() { send quit 2>/dev/null; wait "$ENG_PID" 2>/dev/null; ENG_PID=""; }
show() { sed 's/^/      | /' "$LOG/$1"; }

echo "chessard verify: $ENGINE"

# 1+2 -------------------------------------------------------------------------
start main
send uci
if expect '^uciok'; then
  grep -q '^id name chessard' "$LOG/main.out" && grep -q '^option name Elo type spin' "$LOG/main.out" \
    && pass "uci handshake (id name chessard, Elo option)" || fail "uci handshake incomplete"
  players="$(sed -n 's/^option name Player type combo.*default none //p' "$LOG/main.out")"
  missing=""; for p in carlsen janik kaufman nakamura sadler; do [[ " $players " == *" var $p "* ]] || missing+=" $p"; done
  [ -z "$missing" ] && pass "player adapters listed: ${players//var /}" || fail "players missing:$missing (got: $players)"
else
  fail "no uciok"; show main.out; show main.err
fi
send "setoption name Elo value 2200"
send isready
expect '^readyok' || { fail "no readyok"; show main.err; }
device="$(sed -n 's/.* on \([a-z0-9:]*\)$/\1/p' "$LOG/main.err" | head -1)"
grep -q 'chessard failed to load' "$LOG/main.out" && { fail "model failed to load"; show main.err; }
send "position startpos moves e2e4 d7d5"
t0=$(date +%s.%N)
send go
if expect '^bestmove'; then
  secs=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
  first="$(grep -m1 ' multipv 1 ' "$LOG/main.out")"
  echo "      device: ${device:-?}, move took ${secs}s"
  grep -E '^(info depth|bestmove)' "$LOG/main.out" | sed 's/^/      > /'
  p="$(sed -n 's/.*string p=\([0-9.]*\)%.*/\1/p' <<< "$first")"
  if [[ "$first" =~ ^info\ depth\ 9\ multipv\ 1\ score\ cp\ 91\ .*\ pv\ e4d5\ string ]] \
     && awk -v p="${p:-0}" 'BEGIN{exit !(p >= 93.76 && p <= 93.86)}'; then
    pass "Elo 2200 after 1.e4 d5: depth 9, cp 91, pv e4d5, p=${p}% (expected 93.81 +-0.05)"
  else
    fail "unexpected first line: $first   (expected: info depth 9 multipv 1 score cp 91 ... pv e4d5 string p=93.81%)"
  fi
  [ "$MATCH" = "bestmove e4d5" ] && pass "bestmove e4d5" || fail "got '$MATCH', expected 'bestmove e4d5'"
else
  fail "no bestmove within ${TIMEOUT}s"; show main.out; show main.err
fi
finish

# 3 ---------------------------------------------------------------------------
start kaufman --player kaufman
send uci; expect '^uciok' >/dev/null
send isready
if expect '^readyok' && grep -q 'player: kaufman, Elo 2188' "$LOG/kaufman.err"; then
  pass "--player kaufman: $(grep -m1 'chessard: loaded' "$LOG/kaufman.err" | sed 's/^chessard: //')"
else
  fail "--player kaufman did not log 'player: kaufman, Elo 2188'"; show kaufman.err
fi
send "position startpos moves e2e4 e7e5"
send go
if expect '^bestmove [a-h][1-8][a-h][1-8]'; then pass "--player kaufman plays: $MATCH"
else fail "--player kaufman produced no bestmove"; show kaufman.out; show kaufman.err; fi
finish

# 4 ---------------------------------------------------------------------------
# shellcheck disable=SC2086
$ENGINE --elo 1500 </dev/null >"$LOG/low.out" 2>"$LOG/low.err"; rc=$?
if [ "$rc" = 2 ] && grep -q 'elo must be 2000-2900' "$LOG/low.err"; then
  pass "--elo 1500 rejected (exit 2: $(grep -o 'error: .*' "$LOG/low.err" | head -1))"
else
  fail "--elo 1500 not rejected as expected (exit $rc)"; show low.err
fi

echo
if [ "$FAILS" = 0 ]; then echo "verify: ALL CHECKS PASSED"; exit 0; fi
echo "verify: $FAILS CHECK(S) FAILED"; exit 1
