#!/usr/bin/env bash
# One turn of a headless MadAgents conversation, run detached so it outlives the caller's tool timeout.
#
#   madagents_turn.sh start <madagents_dir> <state_dir> <message_file> [--resume <session_id>]
#                           [--budget <usd>] [--permission-mode <mode>]
#       Starts MadAgents (<madagents_dir>/madagents.sh -p ... --output-format json) in the background.
#       Prints "turn=<NNN>". Files: <state_dir>/turn_<NNN>.{msg,json,err,exit}.
#   madagents_turn.sh wait <state_dir> <NNN> [max_seconds]
#       Blocks up to max_seconds (default 540, i.e. under a 10-min tool timeout).
#       Exit 0 = turn finished (see `show`), 2 = still running, 1 = the claude process exited non-zero.
#   madagents_turn.sh show <state_dir> <NNN>
#       Prints session_id, subtype, cost of this call, sum over all turns, permission denials, and the
#       final message (MadAgents' reply), from turn_<NNN>.json.
#
# Notes
# - --resume <session_id> continues the same MadAgents conversation with its full context.
# - --budget is passed as --max-budget-usd (hard cap for this call). The "sum over all turns" adds the
#   total_cost_usd of every turn's JSON; if the CLI reports a cumulative value on resume, the sum is an
#   upper bound, which is the safe direction for a budget check.
# - Permission prompts cannot be answered in headless mode: a tool call the permission mode does not allow
#   is refused and listed under permission_denials. The approved plan states the mode (default: auto).
set -euo pipefail

usage() { sed -n '2,15p' "$0" >&2; exit 1; }
(($#)) || usage
cmd=$1; shift

case $cmd in
start)
  (($# >= 3)) || usage
  dir=$(realpath "$1"); state=$(realpath -m "$2"); msg=$(realpath "$3"); shift 3
  resume=""; budget=20; pmode=auto
  while (($#)); do
    case $1 in
      --resume) resume=$2; shift 2 ;;
      --budget) budget=$2; shift 2 ;;
      --permission-mode) pmode=$2; shift 2 ;;
      *) echo "unknown option $1" >&2; exit 1 ;;
    esac
  done
  [[ -x $dir/madagents.sh ]] || { echo "ERROR: $dir/madagents.sh not found or not executable" >&2; exit 1; }
  [[ -s $msg ]] || { echo "ERROR: message file $msg is empty" >&2; exit 1; }
  mkdir -p "$state"
  n=$(find "$state" -maxdepth 1 -name 'turn_*.msg' | wc -l)
  k=$(printf %03d $((n + 1)))
  out="$state/turn_$k"
  cp "$msg" "$out.msg"
  args=(-p "$(cat "$msg")" --output-format json --max-budget-usd "$budget" --permission-mode "$pmode")
  [[ -n $resume ]] && args+=(--resume "$resume")
  printf '%s\n' "start=$(date -Is) resume=${resume:-new} budget=$budget permission_mode=$pmode" > "$out.meta"
  (
    cd "$dir"
    OUT="$out" setsid nohup bash -c './madagents.sh "$@" > "$OUT.json" 2> "$OUT.err"; echo $? > "$OUT.exit"' \
      _ "${args[@]}" < /dev/null > /dev/null 2>&1 &
  )
  echo "turn=$k"
  ;;
wait)
  (($# >= 2)) || usage
  state=$1; k=$2; max=${3:-540}; out="$state/turn_$k"
  t=0
  while [[ ! -f $out.exit ]]; do
    ((t >= max)) && { echo "turn $k still running after ${max}s"; exit 2; }
    sleep 15; t=$((t + 15))
  done
  rc=$(cat "$out.exit")
  [[ $rc == 0 ]] && { echo "turn $k finished"; exit 0; }
  echo "turn $k: claude exited $rc; stderr tail:"; tail -20 "$out.err"; exit 1
  ;;
show)
  (($# >= 2)) || usage
  state=$1; k=$2
  python3 - "$state" "$k" <<'PY'
import glob, json, sys
state, k = sys.argv[1], sys.argv[2]
d = json.load(open(f"{state}/turn_{k}.json"))
tot = 0.0
for f in sorted(glob.glob(f"{state}/turn_*.json")):
    try:
        tot += float(json.load(open(f)).get("total_cost_usd") or 0)
    except Exception:
        pass
print("session_id:", d.get("session_id"))
print("subtype:", d.get("subtype"), "| is_error:", d.get("is_error"), "| num_turns:", d.get("num_turns"))
print("cost_this_call_usd: %.4f | sum_all_turns_usd: %.4f" % (float(d.get("total_cost_usd") or 0), tot))
print("permission_denials:", json.dumps(d.get("permission_denials") or []))
print("----- MadAgents reply -----")
print(d.get("result") or "")
PY
  ;;
*) usage ;;
esac
