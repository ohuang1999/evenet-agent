# Physics-analysis pipeline

This project runs physics analyses with agentic AI — a planning-then-approval-gated pipeline of 6 subagents (`physics-planner`, `data-converter`, `fine-tuner`, `predictor`, `data-reconverter`, `result-synthesizer`), defined in `.claude/agents/`, with their supporting templates/scripts in `.claude/agent-resources/<agent-name>/`. The agents plan and drive fine-tuning, inference, and physics-observable extraction around [EveNet](https://arxiv.org/abs/2601.17126), a foundation model for collider-physics event data — EveNet is the underlying model the agents orchestrate, not the thing defining this project's architecture.

**When the user asks for a new physics analysis, follow the pipeline below.** This isn't a skill you invoke by name — it's how you should approach any such request directly.

## Why user interaction is concentrated in one place

Subagents invoked via the `Agent` tool run to completion and return a result — they have no live channel back to the user mid-execution, no way to pose a question and wait for a reply. **Only you (running as the top-level session) can actually talk to the user.** So every question the pipeline needs answered — physics/analysis choices, wall-time, checkpoint override, output naming, everything — is gathered by `physics-planner` into a single plan and resolved in one approval exchange (Step 3/4 below). None of the other five subagents ever ask the user anything; they consume the approved plan as given. If you're ever tempted to have `fine-tuner` or `data-reconverter` "check with the user" about something, that belongs in `physics-planner`'s plan instead — route it there rather than inventing a second interaction point.

## Filesystem scope

Only read, write, or search inside the directory Claude Code was opened in, plus any absolute path the user explicitly gives you (input data location, an existing `EveNet-Full` path, a weights directory, etc.). Never `find`/`ls`/`grep` sibling directories on your own initiative to "discover" things — an existing `EveNet-Full` install, downloaded weights, a reference repo to diff a bug against — even when it would be genuinely useful. If something you need isn't at a path the user has already given you, ask for the path rather than going looking for it yourself. This applies to what you tell subagents to do too — don't hand one a task that requires it to explore beyond paths it's been explicitly given.

**One standing exception: the HEP wiki.** `/global/cfs/cdirs/m2616/ohuang/HEP-agent-memeory/.claude` is a read-only knowledge base (the `agentic_AI` LLM-wiki) that `physics-planner` may consult when planning an analysis. It is in scope for reading and searching — by you and by `physics-planner` — without the user having to name it. It is **never** written to by any agent in this pipeline: no edits, no new pages, no `wiki/log.md` entries. If it isn't present at that path, say so and continue without it; a missing wiki degrades the plan's provenance, it doesn't block the pipeline.

## Step 0: Get the physics request

If not already stated in the conversation, this is the first thing to ask — everything else is plumbing in service of this. Ask the user (one message):
- **What do you want to measure or search for** — the physics prompt `physics-planner` will plan around. Get their actual physics goal, not just "run EveNet."
- **Input data location and format** — absolute path, and whether it's a directory of `.root` files or `.pt` file(s) (some users' events are already saved as PyTorch tensors rather than ROOT ntuples — both are supported, but the format needs to be known upfront since it changes which code path `physics-planner`/`data-converter`/`data-reconverter` use throughout).

Don't ask about tree name (ROOT only), project/run name, slot mapping, or anything analysis-structural here — `physics-planner` derives or proposes those itself (Step 2 below), mostly by inspecting the actual data rather than trusting a description of it.

## Step 1: Establish session context and verify the environment

Two different things here — don't conflate them. **Filesystem state** (is the repo cloned, is the image pulled, are the weights downloaded) is checkable directly and shouldn't be re-asked once true. **Session info** (environment, paths, credentials) can't be detected from the filesystem and has no persistence mechanism in this project — nothing writes it down anywhere — so it has to be confirmed at the start of every session where it isn't already established in the conversation.

Concretely, if not already known from this conversation, ask the user:
- **Environment**: NERSC/Shifter or Docker?
- **EveNet-Full**: ask whether to install a fresh clone or use one they already have — don't assume either way.
  - **Fresh install**: don't ask for a path — follow `setup.md` Step 2 to clone it yourself, then use that resulting path for the rest of the session.
  - **Existing repo**: ask for the absolute path, then validate it rather than trusting it at face value (see filesystem checks below) — a path existing isn't the same as it being a complete, working EveNet-Full checkout.
- **Container image** — default `docker.io/avencast1994/evenet:1.5`; on NERSC this may instead be a `registry.nersc.gov/<project>/avencast/evenet:<tag>` mirror — confirm which, don't assume
- **W&B project name** for this session's runs (and API key, if not already set as an env var — check `echo $WANDB_API_KEY` before asking, don't ask for something already present)
- **NERSC account** (NERSC only, e.g. `m2616`)

The HEP wiki root is fixed (`/global/cfs/cdirs/m2616/ohuang/HEP-agent-memeory/.claude`) — don't ask for it. Just confirm it exists (`ls <wiki>/wiki/index.md`) alongside the other filesystem checks, and note it if absent.

Then check the filesystem, don't assume:
- `EveNet-Full` is actually present at the path in hand (freshly cloned or user-provided) — for a user-provided path, don't stop at "the directory exists": confirm it looks like a real EveNet-Full checkout (e.g. `scripts/train.py`, `evenet/` submodule populated, `share/` templates present) before relying on it, since a wrong or partial path will otherwise surface as a confusing failure much later in the pipeline instead of here
- The `evenet` submodule inside `EveNet-Full` hasn't drifted from what the outer repo's own tree expects (`git -C <evenet_full> submodule status` — no `+`/`-` prefix on the `evenet` line; see `setup.md` Step 2). The outer repo's tree is the authoritative record of which submodule commit goes with which outer-repo commit, so a mismatch here is a real risk, not a formality; this applies equally to a freshly-cloned repo and a user-provided existing one — neither is automatically safe to use
- The container image is actually present (`shifterimg images` on NERSC / `docker images` on Docker) — `physics-planner` will fail its first real step without this, since it needs a working container to inspect the input data (ROOT or `.pt`)
- `Weights/` (at `<parent-of-EveNet-Full>/Weights`, per `setup.md` Step 5 — or wherever the user has explicitly told you the checkpoints live) has both pretrained checkpoints; don't search other directories for them, per the filesystem-scope rule above

If anything is missing, read and follow `.claude/agent-resources/setup/setup.md` to fill the gap (it covers cloning, pulling the image, and downloading weights — skip whichever parts are already done). Don't invoke `physics-planner` against a half-set-up environment and hope for the best.

## Step 2: Invoke physics-planner

Call the `Agent` tool with `subagent_type: "physics-planner"`, `run_in_background: false` (you need its plan before doing anything else — this is exactly the case where waiting in the foreground is correct). Give it:
- The physics prompt and input data location from Step 0
- The session context from Step 1: environment, EveNet-Full path, container image, W&B project, NERSC account
- The HEP wiki root (`/global/cfs/cdirs/m2616/ohuang/HEP-agent-memeory/.claude`), and whether you confirmed it's present — it consults it read-only and cites what it used

It will determine or propose everything else itself: tree name (by inspecting the file — it should list available trees rather than needing one stated, and only escalate to "Open questions" if genuinely ambiguous), project/run name (proposed from the physics goal), slot mapping, targets, split mode, wall time, checkpoint, and output naming — all shown in its plan for you to relay in Step 3.

## Step 3: Present the plan

Show the plan `physics-planner` returned to the user, formatted as it produced it (don't summarize it down — the user needs to see the actual slot mapping, split-mode reasoning, checkpoint/wall-time/output choices, and observable definition to approve meaningfully). If it flagged "Open questions," surface those prominently — they need the user's input before the plan is even complete, separate from the approval decision itself.

## Step 4: Wait for approval

Stop here. Do not invoke `data-converter` or any later subagent in the same turn. This is a hard gate, not a formality. If the user answers open questions or asks for changes, re-invoke `physics-planner` with that feedback to get a revised plan (this is also the mechanism for anything downstream would otherwise have needed to "ask" about — resolve it here, before the pipeline starts, not mid-flight).

**Once approved, save the plan before invoking anything else.** `physics-planner` is an LLM — a later re-invocation on the same prompt isn't guaranteed to reproduce the same plan, so the approved plan text is the only persistent, reproducible record of what was actually decided (head choice, slot mapping, observable definition, everything). Every downstream subagent's numeric output is deterministic given a *fixed* plan (data-converter's split uses a fixed seed; nothing else in the pipeline is stochastic) — the plan itself is the one part that could otherwise only be reconstructed from chat history, which won't always be available. `mkdir -p <evenet_full>/run/<project_name>/` and write the exact approved plan text to `<evenet_full>/run/<project_name>/plan.md` yourself, right here — don't leave it for `data-converter` to do as a side effect of its own setup. The plan's **Wiki consulted** section is part of that record — keep it verbatim, including the wiki commit SHA (`git -C <wiki> rev-parse --short HEAD`, or `no-git`/`unavailable`), so a decision can later be audited against the wiki as it stood at the time rather than as it reads today.

## Step 5: Run the pipeline in order

Once approved, invoke the remaining five subagents **sequentially, in the foreground** (each depends on the previous one's output, so there's no parallelism to exploit here — `run_in_background: false` throughout, except where a subagent's own instructions say to background a specific long-running step like Slurm job monitoring):

1. `data-converter` — pass it the approved plan. Get back: parquet paths, `normalization.pt` path.
2. `fine-tuner` — pass it the plan + data-converter's output. Get back: checkpoint path(s), prediction `.pt` path(s), W&B run URL. (This step includes prediction — see `fine-tuner`'s own notes on why train+predict run as one combined job rather than two.)
3. `predictor` — pass it the plan + fine-tuner's output. Get back: validated prediction `.pt` path(s).
4. `data-reconverter` — pass it the plan + predictor's output. Get back: final output file path (ROOT or `.pt`, matching the plan's input format), match rate.
5. `result-synthesizer` — pass it the plan + data-reconverter's output. Get back: the computed observable and its methodology.

Pass each subagent's full output forward, not just a path — later subagents may need context from earlier steps (e.g. `result-synthesizer` needs the plan's observable definition, not just the output file path).

## Step 6: Final report

Summarize for the user: what was planned, what was approved, what each stage produced (event counts, job IDs/status, match rate), and the final observable with `result-synthesizer`'s stated methodology and its explicit note that the result needs physics review.

## If something fails mid-pipeline

Don't silently retry with different assumptions. Report which stage failed and why (each subagent's own instructions cover diagnosing failures in its domain — e.g. `fine-tuner` reads Slurm error logs, `data-reconverter` investigates a low match rate). If fixing it requires a plan change (not just a bug fix), that's a new plan — go back to Step 3 with the revision and get approval again before resuming.
