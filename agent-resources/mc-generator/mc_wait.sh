#!/usr/bin/env bash
# Wait for MC-generation work without spending model tokens. Meant to be started by the orchestrator with
# Bash run_in_background, so Claude Code re-invokes the orchestrator when it exits.
#
#   mc_wait.sh slurm <poll_seconds> <jobid> [<jobid> ...]
#       Returns when none of the jobs is in squeue any more, then prints a sacct summary
#       (State / ExitCode per job and array task). Exit 0 if every job COMPLETED, 3 otherwise.
#   mc_wait.sh turn <state_dir> <NNN>
#       Returns when a detached MadAgents turn (madagents_turn.sh start) has finished. Exit code = the
#       turn's claude exit code.
set -euo pipefail
(($# >= 2)) || { sed -n '2,11p' "$0" >&2; exit 1; }
mode=$1; shift

case $mode in
slurm)
  poll=$1; shift
  ids=$(IFS=,; echo "$*")
  while [[ -n $(squeue -h -j "$ids" -o %i 2>/dev/null) ]]; do sleep "$poll"; done
  echo "== all jobs left the queue at $(date -Is)"
  sacct -j "$ids" -o JobID%20,JobName%20,State%14,ExitCode,Elapsed,NNodes -X
  bad=$(sacct -j "$ids" -n -X -o State | grep -v -c COMPLETED || true)
  [[ $bad == 0 ]] && exit 0
  echo "== $bad job(s)/task(s) not COMPLETED"; exit 3
  ;;
turn)
  state=$1; k=$2
  while [[ ! -f $state/turn_$k.exit ]]; do sleep 30; done
  rc=$(cat "$state/turn_$k.exit")
  echo "== MadAgents turn $k finished at $(date -Is), exit $rc"
  exit "$rc"
  ;;
*) echo "unknown mode $mode" >&2; exit 1 ;;
esac
