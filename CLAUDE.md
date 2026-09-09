# Physics-analysis pipeline

This project runs physics analyses with agentic AI — a planning-then-approval-gated pipeline of 6 subagents (`physics-planner`, `data-converter`, `fine-tuner`, `predictor`, `data-reconverter`, `result-synthesizer`), defined in `.claude/agents/`, with their supporting templates/scripts in `.claude/agent-resources/<agent-name>/`. The agents plan and drive fine-tuning, inference, and physics-observable extraction around [EveNet](https://arxiv.org/abs/2601.17126), a foundation model for collider-physics event data — EveNet is the underlying model the agents orchestrate, not the thing defining this project's architecture.

**When the user asks for a new physics analysis, follow the pipeline below.** This isn't a skill you invoke by name — it's how you should approach any such request directly.

## Why user interaction is concentrated in one place

Subagents invoked via the `Agent` tool run to completion and return a result — they have no live channel back to the user mid-execution, no way to pose a question and wait for a reply. **Only you (running as the top-level session) can actually talk to the user.** So every question the pipeline needs answered — physics/analysis choices, wall-time, checkpoint override, output naming, everything — is gathered by `physics-planner` into a single plan and resolved in one approval exchange (Step 3/4 below). None of the other five subagents ever ask the user anything; they consume the approved plan as given. If you're ever tempted to have `fine-tuner` or `data-reconverter` "check with the user" about something, that belongs in `physics-planner`'s plan instead — route it there rather than inventing a second interaction point.

## Filesystem scope

Only read, write, or search inside the directory Claude Code was opened in, plus any absolute path the user explicitly gives you (input data location, an existing `EveNet-Full` path, a weights directory, etc.). Never `find`/`ls`/`grep` sibling directories on your own initiative to "discover" things — an existing `EveNet-Full` install, downloaded weights, a reference repo to diff a bug against — even when it would be genuinely useful. If something you need isn't at a path the user has already given you, ask for the path rather than going looking for it yourself. This applies to what you tell subagents to do too — don't hand one a task that requires it to explore beyond paths it's been explicitly given.

**One standing exception: the HEP wiki.** `/global/cfs/cdirs/m2616/ohuang/HEP-agent-memory/.claude` is a read-only knowledge base (the `agentic_AI` LLM-wiki) that `physics-planner` may consult when planning an analysis. It is in scope for reading and searching — by you and by `physics-planner` — without the user having to name it. It is **never** written to by any agent in this pipeline: no edits, no new pages, no `wiki/log.md` entries. If it isn't present at that path, say so and continue without it; a missing wiki degrades the plan's provenance, it doesn't block the pipeline.

## Run isolation: each run is independent

**`<run_dir>` (`<evenet_full>/run/<project_name>/`) is the only run directory that exists for this analysis.** Never `ls`/`find` `<evenet_full>/run/` to see what else is there, and never read, grep, copy from, or cite any other `run/<other_project>/` — not its `plan.md`, not its configs, checkpoints, logs, or outputs — at **any** stage, from planning through the final result. Every analysis is derived fresh from the physics prompt, the input data, the HEP wiki, and the templates in `.claude/agent-resources/` and `EveNet-Full/share/`. Reading your *own* `<run_dir>` is normal and expected (diagnosing a job log, validating a prediction, importing this run's `to_npz.py`) — the rule is about other runs, not your own.

This is what makes a repeat analysis an independent result rather than an echo of the previous one. The two places the leak actually happens are the ends of the pipeline: a planner that reproduces a previous run's slot mapping or selection cut instead of deriving one (this looks like agreement, which is why it's dangerous), and a synthesizer that lands near a number it has already seen. Both are covered by the rule above. It applies to you and to every subagent you invoke — don't hand one a task that requires it to look at another run. **Only exception:** a path the user explicitly gives you, same as `## Filesystem scope`.

## Step 0: Get the physics request

If not already stated in the conversation, this is the first thing to ask — everything else is plumbing in service of this. Ask the user (one message):
- **What do you want to measure or search for** — the physics prompt `physics-planner` will plan around. Get their actual physics goal, not just "run EveNet."
- **Input data location and format** — absolute path, and whether it's a directory of `.root` files, `.pt` file(s), or `.h5` file(s) (some users' events are already saved as PyTorch tensors or HDF5 arrays rather than ROOT ntuples — all three are supported, but the format needs to be known upfront since it changes which code path `physics-planner`/`data-converter`/`data-reconverter` use throughout). Note the output-format asymmetry for h5: an h5-sourced analysis is never reconverted back to h5 — `data-reconverter` always writes `.pt` for h5 input, so downstream analysis works off that `.pt` file rather than a regenerated h5.

Don't ask about tree name (ROOT only), project/run name, slot mapping, or anything analysis-structural here — `physics-planner` derives or proposes those itself (Step 2 below), mostly by inspecting the actual data rather than trusting a description of it.

## Step 1: Establish session context and verify the environment

Two different things here — don't conflate them. **Filesystem state** (is the repo cloned, is the image pulled, are the weights downloaded) is checkable directly and shouldn't be re-asked once true. **Session info** (environment, paths, credentials) can't be detected from the filesystem and has no persistence mechanism in this project — nothing writes it down anywhere — so it has to be confirmed at the start of every session where it isn't already established in the conversation.

Concretely, if not already known from this conversation, ask the user:
- **Environment**: NERSC/Shifter or Docker?
- **EveNet-Full**: ask whether to install a fresh clone or use one they already have — don't assume either way.
  - **Fresh install**: don't ask for a path — follow `setup.md` Step 2 to clone it yourself, then use that resulting path for the rest of the session.
  - **Existing repo**: ask for the absolute path, then validate it rather than trusting it at face value (see filesystem checks below) — a path existing isn't the same as it being a complete, working EveNet-Full checkout.
- **Container image** — default `docker.io/avencast1994/evenet:1.5`; on NERSC this may instead be a `registry.nersc.gov/<project>/avencast/evenet:<tag>` mirror — confirm which, don't assume
- **W&B project name** for this session's runs (and API key, if not already set as an env var — check `echo $WANDB_API_KEY` before asking, don't ask for something already present). **If it's not set, don't accept it pasted into chat and use it ad hoc** — direct the user to `.claude/agent-resources/setup/setup.md`'s W&B step (Step 4) instead: the key needs to be exported in the user's shell *before* launching Claude Code, not provided mid-session, because Claude Code's own permission layer blocks any Bash command containing a literal secret string — and `fine-tuner` specifically needs to run `export WANDB_API_KEY=... && sbatch ...` later, which a chat-provided value can't satisfy without hitting that block.
- **NERSC account** (NERSC only, e.g. `m2616`)

The HEP wiki root is fixed (`/global/cfs/cdirs/m2616/ohuang/HEP-agent-memory/.claude`) — don't ask for it. Just confirm it exists (`ls <wiki>/wiki/index.md`) alongside the other filesystem checks, and note it if absent.

Then check the filesystem, don't assume:
- `EveNet-Full` is actually present at the path in hand (freshly cloned or user-provided) — for a user-provided path, don't stop at "the directory exists": confirm it looks like a real EveNet-Full checkout (e.g. `scripts/train.py`, `evenet/` submodule populated, `share/` templates present) before relying on it, since a wrong or partial path will otherwise surface as a confusing failure much later in the pipeline instead of here
- The `evenet` submodule inside `EveNet-Full` hasn't drifted from what the outer repo's own tree expects (`git -C <evenet_full> submodule status` — no `+`/`-` prefix on the `evenet` line; see `setup.md` Step 2). The outer repo's tree is the authoritative record of which submodule commit goes with which outer-repo commit, so a mismatch here is a real risk, not a formality; this applies equally to a freshly-cloned repo and a user-provided existing one — neither is automatically safe to use
- The container image is actually present (`shifterimg images` on NERSC / `docker images` on Docker) — `physics-planner` will fail its first real step without this, since it needs a working container to inspect the input data (ROOT, `.pt`, or h5)
- `Weights/` (at `<parent-of-EveNet-Full>/Weights`, per `setup.md` Step 5 — or wherever the user has explicitly told you the checkpoints live) has both pretrained checkpoints; don't search other directories for them, per the filesystem-scope rule above

If anything is missing, read and follow `.claude/agent-resources/setup/setup.md` to fill the gap (it covers cloning, pulling the image, and downloading weights — skip whichever parts are already done). Don't invoke `physics-planner` against a half-set-up environment and hope for the best.

## Step 2: Invoke physics-planner

Call the `Agent` tool with `subagent_type: "physics-planner"`, `run_in_background: false` (you need its plan before doing anything else — this is exactly the case where waiting in the foreground is correct). Give it:
- The physics prompt and input data location from Step 0
- The session context from Step 1: environment, EveNet-Full path, container image, W&B project, NERSC account
- The HEP wiki root (`/global/cfs/cdirs/m2616/ohuang/HEP-agent-memory/.claude`), and whether you confirmed it's present — it consults it read-only and cites what it used

It will determine or propose everything else itself: tree name (by inspecting the file — it should list available trees rather than needing one stated, and only escalate to "Open questions" if genuinely ambiguous), project/run name (proposed from the physics goal), slot mapping, targets, split mode, wall time, checkpoint, and output naming — all shown in its plan for you to relay in Step 3.

## Step 3: Present the plan

Show the plan `physics-planner` returned to the user, formatted as it produced it (don't summarize it down — the user needs to see the actual slot mapping, split-mode reasoning, checkpoint/wall-time/output choices, and observable definition to approve meaningfully). If it flagged "Open questions," surface those prominently — they need the user's input before the plan is even complete, separate from the approval decision itself.

## Step 4: Wait for approval

Stop here. Do not invoke `data-converter` or any later subagent in the same turn. This is a hard gate, not a formality. If the user answers open questions or asks for changes, re-invoke `physics-planner` with that feedback to get a revised plan (this is also the mechanism for anything downstream would otherwise have needed to "ask" about — resolve it here, before the pipeline starts, not mid-flight).

**Once approved, save the plan before invoking anything else.** `physics-planner` is an LLM — a later re-invocation on the same prompt isn't guaranteed to reproduce the same plan, so the approved plan text is the only persistent, reproducible record of what was actually decided (head choice, slot mapping, observable definition, everything). Every downstream subagent's numeric output is deterministic given a *fixed* plan (data-converter's split shuffles with the pinned literal seed `42` and reads input files in sorted order, per `data-converter.md` Step 3; nothing else in the pipeline is stochastic — model training itself is seeded by EveNet's own configuration, not by this pipeline) — the plan itself is the one part that could otherwise only be reconstructed from chat history, which won't always be available. **Everything a run generates — plan, YAMLs, job scripts, logs, checkpoints, outputs — stays under `<evenet_full>/run/<project_name>/`; `EveNet-Full/share/` is upstream-tracked reference material and must be read from, never written to.** Generated configs go in `<run_dir>/config/`, and because EveNet resolves a config's `default:` includes relative to that config's own directory (`evenet/control/global_config.py`), those includes must be absolute paths back into `<evenet_full>/share/` — the templates already are. **If `<evenet_full>/run/<project_name>/` already exists, stop and ask the user for a different project name** — an existing directory is a previous run, and per `## Run isolation` it can neither be read nor written into. Don't merge into it, don't clear it, and don't inspect it to decide. Otherwise `mkdir -p <evenet_full>/run/<project_name>/` and write the exact approved plan text to `<evenet_full>/run/<project_name>/plan.md` yourself, right here — don't leave it for `data-converter` to do as a side effect of its own setup. The plan's **Wiki consulted** section is part of that record — keep it verbatim, including the wiki commit SHA (`git -C <wiki> rev-parse --short HEAD`, or `no-git`/`unavailable`), so a decision can later be audited against the wiki as it stood at the time rather than as it reads today.

**At the same time, create `<run_dir>/workflow_state.yaml`** — the persistent record of pipeline progress and recovery history, per `.claude/agent-resources/shared/workflow-state.md`. Seed it with `stage_order: [data_conversion, fine_tuning, prediction, data_reconversion, result_synthesis]`, every stage `not_started`, `current_stage: data_conversion`, `current_status: not_started`, and the `limits` block (`max_attempts_per_stage: 3`, `max_total_recovery_cycles: 5`, `recovery_cycles_used: 0`). Same trust model as `plan.md`: this file is the durable record, not something reconstructed from conversation later.

## Step 5: Run the pipeline in order

Once approved, invoke the remaining five subagents **sequentially, in the foreground** (each depends on the previous one's output, so there's no parallelism to exploit here — `run_in_background: false` throughout, except where a subagent's own instructions say to background a specific long-running step like Slurm job monitoring):

1. `data-converter` — pass it the approved plan. Get back: parquet paths, `normalization.pt` path.
2. `fine-tuner` — pass it the plan + data-converter's output. Get back: checkpoint path(s), prediction `.pt` path(s), W&B run URL. (This step includes prediction — see `fine-tuner`'s own notes on why train+predict run as one combined job rather than two.)
3. `predictor` — pass it the plan + fine-tuner's output. Get back: validated prediction `.pt` path(s).
4. `data-reconverter` — pass it the plan + predictor's output. Get back: final output file path (ROOT or `.pt`, matching the plan's input format), event count (cross-checked against the prediction `.pt`'s own count — there's no match rate to report, since `data-reconverter` reads everything from the prediction `.pt` directly rather than re-matching against the original input).

   **For ROOT-format output, always run this stage** — it's a genuine format conversion nothing else in the pipeline does. **For `.pt`-format output (a `.pt` or h5-sourced plan), you may skip it and pass `predictor`'s validated `predicted.pt` straight to `result-synthesizer` instead**, when `result-synthesizer` can denormalize/decode inline itself (it's instructed to be able to — see `result-synthesizer.md`) and no separate merged/denormalized deliverable file is otherwise wanted. Reconversion's only value-add for `.pt` output is that convenience; going through it anyway adds a translation step that has, in practice, silently dropped a field (`assignment_target`/`assignment_target_mask`) that the native `predicted.pt` already had. If you do skip it, say so plainly in the final report (Step 6) along with why.
5. `result-synthesizer` — pass it the plan + (`data-reconverter`'s output, or `predictor`'s validated `predicted.pt` if step 4 was skipped). Get back: the computed observable and its methodology.

Pass each subagent's full output forward, not just a path — later subagents may need context from earlier steps (e.g. `result-synthesizer` needs the plan's observable definition, not just the output file path).

**After each subagent's normal-mode completion, read its `attempt_N_report.yaml`** (per `.claude/agent-resources/shared/workflow-state.md`) and update `workflow_state.yaml` — mark the stage `completed`, append the attempt to `attempt_history`, advance `current_stage` — before invoking the next subagent. This is what keeps `workflow_state.yaml` an accurate ledger rather than something reconstructed after the fact.

## Step 6: Final report

Summarize for the user: what was planned, what was approved, what each stage produced (event counts, job IDs/status), and the final observable with `result-synthesizer`'s stated methodology and its explicit note that the result needs physics review.

## Failure-directed recovery

Don't silently retry with different assumptions. Every stage failure produces a structured `failure_report.yaml` (per `.claude/agent-resources/shared/workflow-state.md`) — that file, plus `workflow_state.yaml`, is what drives what happens next. This section is what replaced a one-line "report which stage failed and why": that's still the first step, but the record is now a file, not just conversation, and there's now an explicit procedure for repairing an upstream stage rather than restarting the whole pipeline.

**Procedure**:
1. **Read the failing stage's `failure_report.yaml`** in full (`<run_dir>/<stage_id>/attempt_N_failure_report.yaml`).
2. **Classify local vs. upstream** using its `suspected_origin_stage`, `confidence`, and `evidence` fields. **Never re-enter an upstream stage without concrete evidence** — `suspected_origin_stage: self`, `confidence: low` (regardless of what stage is named), or `failure_class: unknown` are all treated the same way: local to the failing stage, or genuinely ambiguous. Only `medium`/`high` confidence naming a specific upstream `stage_id`, backed by concrete `evidence`, justifies re-entering that stage.
3. **If local**: invoke the failing stage's own agent again (normal mode, or its own config-repair path if it has one, e.g. `fine-tuner`'s configuration-class handling) — this is ordinary retry, not upstream recovery.
4. **If upstream-caused**: invoke the upstream agent in **repair mode**, per `.claude/agent-resources/shared/workflow-state.md`'s "Repair/diagnostic-mode entry contract" — hand it explicit paths (the approved plan, `workflow_state.yaml`, its own prior config/manifest/artifacts, the downstream failure report). **Never phrase the task as if the agent remembers a previous invocation** — a fresh subagent call has no memory of one, and even a resumed one shouldn't be relied on to; the whole point of this system is that context comes from files.
5. **After repair, revalidate before retrying downstream.** The repaired stage's own normal validation (already required by its instructions) must pass and be reflected in its new attempt's success report before the originally-failing downstream stage is retried. Don't retry downstream against unvalidated repaired output.
6. **Record every recovery attempt in `workflow_state.yaml`'s `recovery_history`**: trigger stage/attempt, the failure report used, the classified origin stage and confidence, the action taken, the upstream attempt's result, the revalidation outcome, and the downstream retry's result. This is what makes the eventual successful result's provenance reconstructable later.

**Worked example** (the canonical case this system is built around): `fine-tuner` fails and writes `failure_report.yaml` with `suspected_origin_stage: data_conversion`, `confidence: high`, evidence pointing at a specific missing/malformed field → orchestrator reads it, confirms the evidence clears the bar in step 2 → orchestrator invokes `data-converter` in repair mode, passing it the plan, `workflow_state.yaml`, the previous conversion manifest/config, the previous converted output, and `fine-tuner`'s failure report → `data-converter` reads all four, diagnoses, and repairs only if its own inspection confirms the fault is really conversion's → conversion validation is rerun and its new attempt report written → `fine-tuner` is retried against the repaired data.

**Recovery limits** (enforced by the orchestrator, using `workflow_state.yaml`'s `limits` block — no agent enforces this on itself):
- **Max attempts per stage**: `limits.max_attempts_per_stage` (default 3). A stage that has exhausted this stops and escalates to the user rather than trying again.
- **Max total recovery cycles**: `limits.max_total_recovery_cycles` (default 5) across the whole run. Increment `recovery_cycles_used` each time step 4 above actually happens (an upstream repair-mode re-entry); once exhausted, stop and escalate regardless of what any individual stage's own attempt count looks like.
- **No upstream re-entry without evidence** — see step 2 above; this is a hard rule, not a default that can be talked past by a plausible-sounding guess.
- **A repeated identical failure escalates rather than retrying a third way**: if the same stage produces the same `failure_class` + `error_type` on two consecutive attempts (including after a repair), stop and ask the user instead of trying yet another automatic fix — two failures with the same signature means the automatic repair isn't working, not that a third attempt will.
- **`failure_class: unknown` or `confidence: low` always stops and asks the user** — this pipeline doesn't guess its way through an ambiguous failure.

If fixing a failure requires a plan change (not just a bug fix or a repair-mode correction) — a different head, a different cut, a different observable definition — that's a new plan, not a recovery action: go back to Step 3 with the revision and get approval again before resuming, the same as any other plan change.

## Scientific memory, skills, and run/workflow state — keep these separate

Three different things that must not be blurred, especially when extending this pipeline's instructions:
- **Scientific memory**: reusable physics knowledge, observable definitions, analysis conventions — the HEP wiki (`/global/cfs/cdirs/m2616/ohuang/HEP-agent-memory/.claude`), consulted read-only by `physics-planner` only. Never written to by this pipeline.
- **Skills**: reusable procedures, e.g. `.claude/agent-resources/result-synthesizer/unfolding.md`. Consulted where scientifically/procedurally relevant, same as before.
- **Run/workflow state**: the approved plan, generated configs, manifests, converted/predicted/reconverted artifacts, logs, and — as of this section — structured stage/failure reports and `workflow_state.yaml`. This is what failure recovery reads and writes. It is specific to `<run_dir>`, never shared across runs (per `## Run isolation`), and never a substitute for scientific memory or a skill — a repair decision is made from run state and the relevant skill/plan section, not from the wiki, unless the failure is itself a scientific/methodology question (in which case that's a plan revision, handled by re-invoking `physics-planner`, not a recovery action).
