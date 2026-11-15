# Workflow state and failure-recovery conventions

Shared reference for every post-approval pipeline agent (`data-converter`, `fine-tuner`, `predictor`, `data-reconverter`, `result-synthesizer`) and for the top-level orchestrator (`CLAUDE.md`'s "Failure-directed recovery" section). Defines the one schema and taxonomy everyone uses, so it's written once here rather than re-stated per agent.

**The core principle**: agent invocations are stateless — each one runs once and returns. The run itself is stateful through explicit files in `<run_dir>`. A restarted or repair-mode-re-entered agent reconstructs its context by reading those files, never by assuming it (or the orchestrator) remembers a previous invocation.

`physics-planner` is out of scope for everything below — it's pre-approval and already has its own persistence (`plan.md`, revised by re-invocation with feedback per `CLAUDE.md` Step 4).

## Stage IDs

One per post-approval agent, used as the directory name for that stage's reports (distinct from the agent name, matching the "Phase 1/2/3" language already used in the agent files):

| Stage ID | Agent | Phase |
|---|---|---|
| `data_conversion` | `data-converter` | Phase 1 |
| `fine_tuning` | `fine-tuner` | Phase 2 |
| `prediction` | `predictor` | (validates Phase 2's predict output, or runs its own) |
| `data_reconversion` | `data-reconverter` | Phase 3 |
| `result_synthesis` | `result-synthesizer` | — |

## Report file locations

A dedicated namespace, `<run_dir>/<stage_id>/`, separate from the existing functional directories (`data_processed/`, `ckpts/`, `predict/`, `output/`) so reports never collide with data artifacts:

- Success: `<run_dir>/<stage_id>/attempt_<N>_report.yaml`
- Failure: `<run_dir>/<stage_id>/attempt_<N>_failure_report.yaml`
- Repair-mode, checked-and-not-at-fault: `<run_dir>/<stage_id>/attempt_<N>_diagnostic_report.yaml`

`<N>` is a 1-based counter, incrementing per attempt at that stage (a normal run and every subsequent repair-triggered re-entry each get the next number) — read `workflow_state.yaml`'s `stages.<stage_id>.attempts` to know the next value; don't guess or overwrite a prior attempt's file.

## Success report schema

Facts, provenance, and validation results only — **never chain-of-thought or internal reasoning**. Write this at the end of every normal-mode run, in addition to the prose summary you return to the orchestrator.

```yaml
stage: <stage_id>
attempt: <N>
status: completed
started: <ISO8601>
finished: <ISO8601>
inputs_used:
  plan: <path to plan.md>
  # ...other upstream artifact paths actually consumed...
outputs_produced:
  # ...named paths this stage produced...
decisions:
  - "...a key config value, threshold, or convention actually applied (not why you considered it)..."
validation:
  - {check: "...", result: pass|fail}
artifact_paths:
  - ...
```

For a repair-mode attempt, add one more block (see "Repair/diagnostic-mode entry contract" below):

```yaml
repair_context:
  triggered_by_stage: <stage_id of the stage whose failure caused this re-entry>
  triggered_by_failure_report: <path>
  diagnosis: "..."
  changed: "...exactly what was modified..."
  unchanged: "...what was verified untouched..."
```

## Failure report schema

Write this instead of a success report when a stage fails. Fields:

```yaml
stage: <stage_id>
status: failed
failure_class: input_validation | configuration | infrastructure | convergence | downstream_validation | unknown
error_type: <short code, e.g. "missing_extra_save_key", "oom", "slurm_node_fault">
error_summary: <one paragraph>
observed: <what actually happened>
expected: <what should have happened>
affected_artifacts: [...]
suspected_origin_stage: <stage_id> | self | unknown
confidence: high | medium | low
evidence: [...concrete facts backing suspected_origin_stage — a specific missing key, a shape mismatch, a log line...]
recommended_action: <text>
retryable: true | false
attempt: <N>
```

**Failure taxonomy** — pick the one that actually fits; use `unknown` rather than forcing a fit:
- `input_validation` — the stage's input data doesn't have what was expected (missing field, wrong shape/dtype, an invariant violated).
- `configuration` — a YAML/config value is wrong, missing, or inconsistent with another config (e.g. finetune vs. predict head mismatch).
- `infrastructure` — Slurm/scheduler/node/container problem unrelated to the analysis itself.
- `convergence` — training didn't converge (fine-tuner-specific; see its own Step 6, unchanged by this doc).
- `downstream_validation` — this stage's own output failed a validation check that stage itself runs (not caused by an upstream stage).
- `unknown` — doesn't clearly fit the above, or the evidence is too thin to be confident. Always escalate rather than guess a repair.

`suspected_origin_stage` and `confidence` are what the orchestrator uses to decide whether to re-enter an upstream agent (see `CLAUDE.md`'s "Failure-directed recovery") — state them honestly. `self` with `input_validation`/`configuration`/`downstream_validation` means "my own stage's problem, no upstream re-entry needed." Naming an upstream `stage_id` with `confidence: low` is treated by the orchestrator the same as `unknown` — it does not trigger upstream re-entry on its own; only `medium`/`high` confidence with concrete `evidence` does.

## Repair/diagnostic-mode entry contract

When the orchestrator re-enters a stage in repair mode, it always hands the agent **explicit file paths**, never a request to "recall" a prior invocation:
- the approved plan (`<run_dir>/plan.md`)
- `<run_dir>/workflow_state.yaml`
- this stage's own most recent attempt's config/manifest and produced artifacts (from `workflow_state.yaml`'s `attempt_history`)
- the downstream failure report that triggered re-entry

The agent reads all of these from disk before doing anything else. It then either:
- **(a) genuinely at fault**: makes the minimal necessary fix (not a full redo unless actually required), reruns its own normal-mode validation, and writes a new attempt's success report with `repair_context` filled in recording exactly what changed and what didn't; or
- **(b) not at fault**: changes nothing, and writes a diagnostic report stating its own artifacts passed validation and citing the evidence, so the orchestrator looks elsewhere.

`data-converter.md` has the full worked-out version of this contract; `predictor.md`, `data-reconverter.md`, and `result-synthesizer.md` reference this section directly rather than repeating it.
