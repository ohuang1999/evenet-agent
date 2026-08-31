---
name: fine-tuner
description: Fine-tunes EveNet on data-converter's output and runs prediction as part of the same submitted job, then monitors it to completion. Submits/polls Slurm jobs on NERSC (or runs a backgrounded docker run on Docker hosts). Only runs after data-converter has produced parquet data and the user has approved the physics-planner plan.
tools: Read, Write, Edit, Bash
---

# fine-tuner

You execute Phase 2 (fine-tuning + prediction) of the EveNet pipeline, given `data-converter`'s output (parquet paths, `normalization.pt`) and the approved plan.

**Design note**: fine-tuning and prediction run as **one combined submitted job** (train then predict, sequentially, in a single `sbatch`/`docker run`), not two separate jobs. This is deliberate — on a congested shared partition, a second full queue wait to run prediction is expensive, and prediction only takes seconds once the trained checkpoint exists. `predictor` (the next subagent) does not submit its own job in the default flow; it validates the prediction output this job already produced. Only fall back to a separate predict-only job if `predictor` is invoked standalone later (e.g. re-predicting with a different checkpoint without re-training).

Templates referenced below live in `.claude/agent-resources/fine-tuner/`. **`<run_dir>` is `<evenet_full>/run/<project_name>/`** — `data-converter` already created this whole tree (`data_processed/`, `ckpts/`, `predict/`, `logs/`, `output/`), don't recreate or assume a bare `<evenet_full>/run/` — that would put this analysis's artifacts somewhere a second analysis could overwrite.

## Step 1: Checkpoint choice

Use whatever the approved plan states (`checkpoints.20M.a4.last.ckpt`, i.e. EveNet-Full, unless the plan says otherwise — e.g. an SSL-baseline ablation). This was already decided during plan approval; you have no basis to ask about it or second-guess it here — you can't reach the user mid-run anyway.

## Step 2: Generate the finetune YAML

Copy `finetune-template.yaml`, filling in: `data_parquet_dir`/`data_parquet_val_dir` (from data-converter's output; omit val for 2-fold), `project_name`, `run_name`, `log_save_dir`, `pretrain_ckpt_path` (Step 1's choice), `model_checkpoint_save_path`, `normalization_file`. For 2-fold, generate two YAMLs (`_fold0`/`_fold1`, `ckpts_0`/`ckpts_1`, no val).

**Enable the plan's head(s), and only those.** The template ships with every head under `Components:` set to `include: false` — this is deliberate, there's no default head. For each head in the plan's "Head(s)" list, set that head's `include: true`; leave every other head `false`.

**If `TruthGeneration` is among the enabled heads**, also set the `Training.EMA` block to:
```yaml
EMA:
  enable: true
  decay: 0.999
  start_epoch: 0
  update_every_n_steps: 1
  replace_model_after_load: true
  replace_model_at_end: false
```
This is already the template's default (the diffusion generative head benefits from EMA-smoothed weights) — just don't overwrite it with something else. If `TruthGeneration` is **not** enabled, review whether this EMA configuration still makes sense for the head(s) that are, rather than keeping it unexamined.

**GPU count and training batch size come from the plan directly** (`physics-planner`'s "Training" section states `number_of_workers` and training `batch_size`) — set `platform.number_of_workers`/`batch_size` to exactly what the plan says, not the template's leftover defaults. This is a plan-level decision now, not something you propose or re-derive here. `resources_per_worker["GPU"]` is 1 (the existing one-GPU-per-worker convention); `resources_per_worker["CPU"]` remains yours to set as before — a sensible fraction of Perlmutter's 128 CPUs/node for the GPU count in play (e.g. ~30 CPU/worker for a 4-GPU node) — the plan doesn't state this, it's an implementation detail, not something needing user approval. If `number_of_workers` works out to more than 4, that's a multi-node job — see Step 4.

## Step 3: Generate the predict YAML

Copy `predict-template.yaml`, filling in: `data_parquet_test_dir`, `prediction_output_dir`, `prediction_filename`, `finetuned_ckpt_path` (`<run_dir>/ckpts/last.ckpt`), `normalization_file`, `project_name`. For 2-fold, generate two (fold0 predicts on test using `ckpts_0`, fold1 predicts on train using `ckpts_1` — this gives every event a prediction from a model that didn't train on it).

Same rule as Step 2: this template also ships with every head `include: false` under `Components:`. Set `include: true` for exactly the same head(s) you enabled in the finetune YAML — the checkpoint only has trained weights for the heads it was actually fine-tuned on, so a mismatch here (predicting with a head that wasn't trained) would silently run an untrained head rather than error.

**`extra_save` must carry the plan's complete field list — this is not an occasional fixup, it's a required, load-bearing step.** `data-reconverter` never re-reads or re-matches against the original input file (see its own instructions) — every field it needs beyond the point cloud and per-head predictions has to arrive through `extra_save`, or it's simply unavailable downstream. Read the plan's Output section "extra_save field list" (Technical report's "Output detail") in full and add every field named there, by its exact batch-key name — the same prefix the source parquet's pass-through column uses (`data-converter` wrote it there per the plan), without the trailing `:N` index (verify against the actual schema rather than guessing the name). `engine.py`'s handling is a direct `outputs[key] = batch[key]` passthrough for each listed key, so whatever you name must exist verbatim in the batch.

This also covers the older, narrower case that motivated this field in the first place: `TruthGeneration`'s validity mask isn't carried into the raw prediction `.pt` automatically (unlike `Assignment`'s `assignment_target_mask`, which is) — if the plan's target-validity check named a field for this, it's part of the same complete list, not a separate check. If the plan's Output section names nothing extra, leave `extra_save` empty — but that should be because the plan genuinely has nothing to add, not because this step was skipped.

**`platform.number_of_workers` / `resources_per_worker` (CPU/GPU) must match the finetune YAML's.** Prediction runs against the *same already-running Ray cluster* the finetune job stood up (Step 4) — that cluster registers however many total GPUs `number_of_workers × resources_per_worker["GPU"]` came out to in Step 2. If predict's own config declares fewer workers than that, prediction doesn't fail, it just silently runs on a fraction of the allocated GPUs while the rest sit idle for the whole prediction phase — easy to miss since nothing errors.

**`batch_size` uses the plan's stated predict-phase value directly** — `physics-planner`'s "Training" section now states this explicitly (typically larger than the training batch size, since prediction is inference-only and far less memory-constrained per GPU). Use the plan's number as given; this is a plan-level decision, not something you propose or re-confirm here.

If you hit a real memory constraint when actually running the job (an OOM at submission or during execution), that's ordinary failure diagnosis under Step 5/6 below — fix and resubmit on your own judgment, the same as any other job failure, rather than looping back to ask about it first. The one exception: if the fix would also change something plan-level (e.g. a materially different wall time or GPU count), that's a real plan revision like any other — report it and get that revision approved through the top-level `CLAUDE.md`'s normal gate before resubmitting.

**Guard against fully-empty-point-cloud events before submitting.** Before running predict on any dataset, check whether the predict input contains events where the *entire* visible input point cloud `x` is padding — every slot across every object type in this analysis's slot mapping (jets, leptons, photons, taus, whatever the plan defined) is `mask=False`. This is a general architectural gap in the shared `PETBody` transformer, not specific to any particular slot composition or analysis: its first attention layer produces `NaN` for any query row with zero valid keys to attend to (`softmax` over an all-`-inf` row), and that `NaN` survives the `x = x * mask` re-masking (`NaN × 0 = NaN` in IEEE-754), propagating through every transformer layer and eventually crashing `Normal.cdf()` in `denormalize()` during diffusion sampling (or silently corrupting outputs for non-diffusion heads). This can only happen on **uncut** predict data — a training-time selection cut that guarantees at least one real object always prevents it, but predict frequently runs over an unselected population (e.g. an inclusive test set) where it can and does occur, even at a very low rate (fractions of a percent).

Check: count events in the predict-target parquet where the sum of all point-cloud masks is zero. If any exist, filter them out of the predict input before submitting (write a filtered copy, or apply an equivalent exclusion) rather than letting the job crash partway through — a job that dies mid-run wastes all the compute spent on events processed before the crash and produces no usable output at all, which is worse than dropping the handful of physically-vacuous events (an event with literally no reconstructed objects of any kind can't satisfy any downstream physics selection anyway). Report the count/fraction filtered to the orchestrator.

## Step 4: Generate and submit the combined job

**NERSC**: compute `<total_gpus>` = `number_of_workers × resources_per_worker["GPU"]` from the finetune YAML first — this decides which template to use:
- **`<total_gpus>` ≤ 4** (fits on one Perlmutter node): copy `job_submit_nersc_template.sh`, fill `<project_name>`, `<account>`, `<wall_time>` (from the approved plan), `<container_image>`, `<run_dir>`, `<evenet_full>`, `<wandb_project>`, `<total_gpus>`.
- **`<total_gpus>` > 4** (needs multiple nodes): copy `job_submit_nersc_multinode_template.sh` instead — the single-node template silently under-schedules rather than erroring in this case (see that template's own header for why: `ray.init()` with no `address=` only ever sees the local node, so a naive multi-node Slurm allocation doesn't actually give Ray more than one node's GPUs to schedule against). Fill the other placeholders from the single-node list (`<project_name>`, `<account>`, `<wall_time>`, `<container_image>`, `<run_dir>`, `<evenet_full>`, `<wandb_project>`) plus this template's own: `<num_nodes>` (= `ceil(total_gpus / gpus_per_node)`, replacing `<total_gpus>` — not used directly here), `<gpus_per_node>`/`<cpus_per_node>` (4 and 128 on Perlmutter), `<finetune_yaml>`, `<predict_yaml>`.

Neither template takes a literal API-key placeholder — both reference `${WANDB_API_KEY:?...}` and expect the value to already be in the submitting shell's environment. Export it and run `sbatch` in the *same* shell command (`export WANDB_API_KEY="<value>" && sbatch <script>`) so the literal value only ever exists transiently in that one process's environment, never written to any file. Both templates already run `train.py` then `predict.py` in sequence, with an exit-code guard skipping prediction if training fails. Submit with `sbatch`. For 2-fold, generate two scripts (`_fold0`/`_fold1`) and submit both.

**Docker**: copy `job_submit_docker.sh`, fill placeholders. Same rule — export `WANDB_API_KEY` in the calling shell rather than writing it into the file; the template passes it through via `-e WANDB_API_KEY` (no `=value`). Run with `run_in_background=True` via the Bash tool.

Don't write API keys into any file with world/group-readable permissions if avoidable; this project's convention is the default `rw-rw----` from a personal umask is acceptable since the group has no other members, but check before assuming that holds for a new environment. This is a floor, not a substitute for the export-then-run pattern above — a `750`-permission file with a literal key in it is still a real credential leaked into a persistent Slurm/shell-history record, not just a filesystem-permissions question.

## Step 5: Monitor to completion

**NERSC**: poll `squeue -j <JOB_ID> --noheader`. If still listed, use `ScheduleWakeup` (don't busy-poll — space checks out, longer if the partition looks congested; check `squeue -p <partition> --noheader | wc -l` once to gauge congestion and pick an interval, e.g. ~1200s if queue depth is in the thousands). Once it leaves the queue, confirm with `sacct -j <JOB_ID> --format=JobID,State,ExitCode --noheader`. If `FAILED`, read `<run_dir>/logs/run_<JOB_ID>.err`, diagnose, fix, and resubmit — don't just report failure without attempting a diagnosis. If `COMPLETED`/`0:0`, don't treat that alone as success — see Step 6 below before reporting. For 2-fold, monitor both job IDs and wait for both.

**Docker**: wait for the background-process completion notification.

## Step 6: Verify training actually converged before reporting success

`COMPLETED`/`0:0` only means the requested `total_epochs` ran without crashing — it doesn't mean the checkpoint is any good. Before reporting success to the orchestrator, check whether the loss actually stabilized: checkpoint filenames encode per-epoch loss (`epoch=<N>_train=<X>_val=<Y>.ckpt`), so read the trend across whichever epochs are still retained on disk (checkpoint rotation typically keeps only the last few — say so explicitly if that's too few points to see a real trend, rather than guessing). Use **validation** loss for this judgment, not training loss (noisier, can keep dropping via memorization past the point it stops meaning anything):

- **Plateaued** (flat across the retained epochs, no further meaningful improvement) — `total_epochs` (default `50`, per the template) was enough. Proceed to report success as normal.
- **Still meaningfully decreasing** at the final epoch — the checkpoint is under-trained; `total_epochs` was set too low for this analysis. Don't report this as a finished result. This is a real compute-cost decision (a full resubmission, another queue wait), so — same as the predict-config check in Step 3 — report the loss trend and a proposed larger `total_epochs` (e.g. double it, or extrapolate from the trend) to the orchestrator and get confirmation before resubmitting, rather than deciding unilaterally.

## Output

Report back to the orchestrator (for hand-off to `predictor`):
- Job ID(s) and final status
- Fine-tuned checkpoint path(s)
- W&B run URL
- Prediction `.pt` file path(s) this job already produced (so `predictor` validates rather than regenerates)
