# EveNet environment setup

One-time setup so the environment is ready for `physics-planner` and the rest of the agent pipeline. This does not require GPU or launching a container. Read and follow this directly (it's a plain instructions file, not a discovered skill or agent) whenever the environment looks unset up.

## Step 1: Ask the user's computing environment

Ask the user whether they are on:
- **NERSC / HPC with Shifter**
- **A server with Docker available**

Record the environment type for use by the rest of the pipeline.

## Step 2: Clone the EveNet repository

```bash
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
cd EveNet-Full
```

**No manual pin.** Use `main` for the outer repo, and whatever commit the outer repo's own tree references for the `evenet` submodule (i.e. plain `--recursive`/`git submodule update --init`, not a separately-chosen submodule commit) — the submodule pointer inside the outer repo's tree is the authoritative pairing, so don't override it with an independently-chosen `evenet` commit. Verify this pairing rather than assuming it:

```bash
git -C EveNet-Full submodule status
```

No `+` (or `-`) prefix on the `evenet` line means the checked-out submodule commit matches what the outer repo's tree expects — safe. A `+` prefix means they've drifted apart (e.g. someone `git checkout`'d the submodule to a different commit by hand without updating the outer repo, or vice versa) — run `git submodule update --init` to snap it back to what the outer repo's tree actually pins, rather than leaving the mismatch or guessing which side is "right."

A checkpoint fine-tuned under one `evenet` commit may not load under a different one if the model architecture changed in between (missing/mismatched-shape keys in the state dict). If that happens, retrain rather than expect a code-level fix to bridge the two versions.

If the repo is already cloned, don't assume it's safe — run the `submodule status` check above before trusting it, and pull `main` on the outer repo too if it looks stale (check `git -C EveNet-Full fetch origin main && git -C EveNet-Full log --oneline HEAD..origin/main` — anything listed means you're behind).

## Step 3: Pull the container image

**Docker:**
```bash
docker pull docker.io/avencast1994/evenet:1.5
```

**NERSC (Shifter):**
```bash
shifterimg pull docker.io/avencast1994/evenet:1.5
```

## Step 4: Configure W&B credentials

**Check first, don't ask blind**: `echo $WANDB_API_KEY` — if it's already set, it's already usable, skip straight to asking for `WANDB_PROJECT` only.

**If it's not set, don't accept it typed into chat and use it ad hoc.** `WANDB_API_KEY` needs to already be in the environment the Claude Code session itself was launched with, not something provided mid-session — every Bash command this pipeline runs is subject to a permission-layer block on literal secrets appearing in the command text, so a key that exists only as chat text (not as an inherited env var) can't actually be used later where it's needed: `fine-tuner` submits jobs via `export WANDB_API_KEY=... && sbatch ...` in one command, which hits this block if the value has to be typed into it. Tell the user to export it in their shell — ideally a persistent profile like `~/.bashrc` so it survives across sessions — and start a new Claude Code session, rather than pasting the raw value into the conversation. See the top-level `README.md`'s "Install" section for the exact steps; point the user there rather than re-explaining it here. The mechanism: the user's shell exports it before launching `claude`, the `claude` process inherits it, and every subprocess it spawns (including all Bash tool calls) inherits it too via normal process environment inheritance — so the literal value never needs to appear in any command text.

Also ask for:
- `WANDB_PROJECT`: A descriptive project name chosen by the user.

Record the project name for use by the rest of the pipeline. Never write the API key itself to any file, committed or not.

## Step 5: Download pretrained weights

Download checkpoint files from HuggingFace into a `Weights` folder at a fixed, predictable location — next to `EveNet-Full` (i.e. `<parent-of-EveNet-Full>/Weights`), not "wherever's convenient":

```bash
mkdir -p Weights
wget -O Weights/SSL.20M.last.ckpt \
  "https://huggingface.co/Avencast/EveNet/resolve/main/SSL.20M.last.ckpt"
wget -O Weights/checkpoints.20M.a4.last.ckpt \
  "https://huggingface.co/Avencast/EveNet/resolve/main/checkpoints.20M.a4.last.ckpt"
```

If checkpoints already exist at that location, verify their presence and skip download. If they're not there, don't go searching other directories for them (see `CLAUDE.md`'s filesystem-scope rule) — ask the user for the absolute path if they say they already have them somewhere else.

Both are downloaded because `physics-planner`/`fine-tuner` need to pick one as `pretrain_ckpt_path`:
- `checkpoints.20M.a4.last.ckpt` is **EveNet-Full** (Stage 2: backbone + jointly pretrained Classification + TruthGeneration + ReconGeneration heads). Recommended default — the EveNet paper shows it consistently outperforms SSL-only as a fine-tuning starting point, including on decay topologies unseen during pretraining.
- `SSL.20M.last.ckpt` is **EveNet-SSL** (Stage 1: backbone + ReconGeneration only, self-supervised, no Classification/TruthGeneration heads). Weaker as a starting point per the paper, but a valid choice for a from-scratch-head ablation/baseline.

`physics-planner` states this default in its plan rather than asking proactively — see the top-level `CLAUDE.md` for how the pipeline is structured.

## Output

After completing all steps, report:
- Environment type (Docker / NERSC)
- EveNet-Full repo path
- Weights directory path with available checkpoints
- Confirmation that the environment is ready for `physics-planner`
