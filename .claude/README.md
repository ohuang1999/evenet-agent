# EveNet Physics-Analysis Agents

An agentic AI pipeline for [Claude Code](https://claude.com/claude-code) that plans and runs physics analyses on top of [EveNet](https://arxiv.org/abs/2601.17126), a foundation model for collider-physics event data. Six coordinated subagents take a plain-language physics request through planning, data conversion, fine-tuning, prediction, export, and observable computation — with a single, explicit approval gate before anything actually runs.

- **EveNet paper**: [arXiv:2601.17126](https://arxiv.org/abs/2601.17126)
- **EveNet docs**: https://evenet-hep.github.io/EveNet-Full/
- **EveNet-Full repo**: https://github.com/EveNet-HEP/EveNet-Full

## Install

This repo's own root is `.claude/` — clone it directly there, not into a subfolder (`CLAUDE.md`, `agents/`, and `agent-resources/` all need to sit immediately under `.claude/` for Claude Code to find them):

```bash
mkdir -p .claude
git clone <this-repo-url> .claude
```

`.claude/CLAUDE.md` is auto-loaded as project context every session — no invocation needed. `.claude/agents/*.md` are auto-discovered as custom subagents (each file must be a direct child of `agents/`, not nested).

## Usage

Just describe the physics analysis you want:

> "I want to measure the spin-density matrix elements of the photon in Bs1 → B* γ decays using my ROOT ntuples at /path/to/data"

Claude will walk the pipeline below automatically. You'll see one plan to review and approve; everything else runs unattended from there.

## How it works

```
                    ┌─────────────────┐
   physics prompt → │  physics-planner │ → plan (head choice, slot mapping,
   + input data     │   (Agent tool)   │    targets, split mode, training/output
                     └─────────────────┘    settings, observable definition)
                              │
                    ┌─────────▼─────────┐
                    │  present the plan  │
                    │  wait for approval │ ◄── hard gate, not a formality
                    └─────────┬─────────┘
                              │ approved
                    ┌─────────▼─────────┐
                    │  save plan.md      │  (reproducibility record — physics-planner
                    └─────────┬─────────┘   is an LLM, re-running it isn't guaranteed
                              │              to reproduce the same plan)
        ┌─────────────────────┼─────────────────────────────────────┐
        ▼                     ▼                     ▼               ▼            ▼
 data-converter  →   fine-tuner       →   predictor    →  data-reconverter → result-synthesizer
 (ROOT/.pt → npz/    (train+predict,        (validates       (predictions →      (computes the
  parquet)            one combined job)      prediction)      ROOT/.pt output)    observable)
```

**Only `physics-planner` talks to you.** Subagents invoked via the `Agent` tool run to completion and return a result — they have no live channel back to you mid-execution. So every question the pipeline needs answered (physics choices, wall-time, checkpoint override, output naming — everything) is gathered into `physics-planner`'s single plan and resolved in one approval exchange. None of the other five subagents ever ask you anything; they execute whatever the approved plan says. See `CLAUDE.md` for the full step-by-step orchestration logic.

## The six subagents

| Agent | Does |
|---|---|
| `physics-planner` | Inspects the actual input data (never trusts a description of it), proposes head(s), slot mapping, targets, split mode, and everything else downstream needs. Read-only — never executes anything. |
| `data-converter` | Converts ROOT or `.pt` input into EveNet's point-cloud npz/parquet format per the approved plan. |
| `fine-tuner` | Generates the finetune/predict YAMLs, submits fine-tuning + prediction as one combined NERSC/Docker job, monitors it to completion. |
| `predictor` | Validates the prediction `.pt` output (or runs a standalone prediction if invoked without an upstream `fine-tuner` job). |
| `data-reconverter` | Matches predictions back to their original events and writes output in the *same format* the input was (ROOT in → ROOT out, `.pt` in → `.pt` out). |
| `result-synthesizer` | Computes the plan's specified physics observable from the final output — a measurement (e.g. spin-density matrix elements) or a discrimination metric (e.g. Significance Improvement Characteristic). |

## Supported heads

| Head | Priority |
|---|---|
| `TruthGeneration` | Primary — full support, including target/invisible-particle definition |
| `Assignment` | Primary — full support, including resonance decay-chain topology |
| `Classification` | Supported if you define real category labels; otherwise stays present-but-disabled |
| `ReconGeneration`, `GlobalGeneration` | Supported, but low priority — only propose these when the goal specifically calls for them |
| `Segmentation`, `Regression` | **Not yet supported** — `physics-planner` will flag rather than guess if an analysis seems to need one |

`physics-planner` can select more than one head per analysis (e.g. `TruthGeneration` + `Assignment` to predict an invisible particle within a specific reconstructed resonance).

## Input and output formats

Both ROOT (`.root` ntuples) and `.pt` (PyTorch tensor files, for users without ROOT data) are supported as input. `data-reconverter`'s output always mirrors the input format — predictions get attached to whatever you actually gave the pipeline, not converted to a different format along the way.

## What's in this repo

```
CLAUDE.md                          Orchestration logic — auto-loaded every session
agents/
  physics-planner.md               Plans; read-only; the only agent that "talks" to you
  data-converter.md                ROOT/.pt -> npz/parquet
  fine-tuner.md                    Fine-tuning + prediction (combined Slurm/Docker job)
  predictor.md                     Validates prediction output
  data-reconverter.md              Predictions -> ROOT/.pt (matching input format)
  result-synthesizer.md            Computes the physics observable
agent-resources/
  setup/setup.md                   One-time environment setup (repo clone, container, W&B, weights)
  data-converter/                  event_info template, data-prep scripts
  fine-tuner/                      finetune/predict templates, job-submission scripts
  predictor/                       predict template (standalone-mode fallback)
  data-reconverter/                export scripts
```

## Things worth knowing before you dig in

- **`<run_dir>` is always `<evenet_full>/run/<project_name>/`**, never bare `<evenet_full>/run/` — every agent's artifacts (converted data, checkpoints, predictions, logs) live under a project-specific subdirectory so a second analysis can't silently overwrite a first one's.
- **Checkpoint default is `checkpoints.20M.a4.last.ckpt` (EveNet-Full, Stage 2)**, not the SSL-only checkpoint — the EveNet paper shows it consistently outperforms SSL-only as a fine-tuning start, including out-of-distribution. `physics-planner` states this as the default; override it in the plan if you want an SSL-baseline ablation.
- **Global conditions are always declared as a fixed 10-feature schema** (`met`, `met_phi`, `nLepton`, `nbJet`, `nJet`, `HT`, `HT_lep`, `M_all`, `M_leps`, `M_bjets`) matching the pretraining corpus, even for analyses with none of these. This isn't optional: an empty `Conditions: {}` produces a zero-length feature list that crashes model construction (`torch.tensor([])` defaults to `float32`, not `bool`, breaking `torch.where` in EveNet's `Normalizer`) — hit this in production once. Matching the schema also lets the pretrained checkpoint's `GlobalEmbedding` weights actually load instead of being silently skipped for a shape mismatch.
- **Task tensors are omitted, not zero-filled, for heads you didn't select.** EveNet's preprocessing treats each task's tensors as present-or-absent — a task with nothing in the npz is just logged "inactive." This is documented behavior, not a workaround.
- **Reproducibility**: the approved plan (head choice, slot mapping, observable definition) is saved to `<run_dir>/plan.md` right after approval, since `physics-planner` is an LLM and a later re-invocation isn't guaranteed to reproduce the same plan. Everything downstream is deterministic given a fixed plan (the train/val/test split uses a fixed seed; nothing else in the pipeline is stochastic) — but EveNet's own training loop has no seed/determinism control, so bit-identical retraining isn't guaranteed even from an identical plan.

## Requirements

- Git (to clone `EveNet-Full` and this repo)
- NERSC account with Shifter, or a Docker host with GPU access
- A [Weights & Biases](https://wandb.ai/) account and API key
