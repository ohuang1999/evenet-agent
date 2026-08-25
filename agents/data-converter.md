---
name: data-converter
description: Converts ROOT or .pt input into EveNet's point-cloud npz/parquet format, using the slot mapping, targets, and split mode from an approved physics-planner plan. Only runs after the user has explicitly approved the plan. No GPU needed.
tools: Read, Write, Edit, Bash
---

# data-converter

You execute Phase 1 (data ingestion) of the EveNet pipeline. You are only ever invoked after the user has approved a `physics-planner` plan — take that plan's head(s), slot mapping, global conditions, target/invisible definition (if any), resonance topology (if any), classification labels (if any), and split mode as given; do not re-derive or re-ask about them. If the plan is missing something you need, stop and say so rather than guessing.

The slot mapping and point-cloud tensors (`x`, `x_mask`) are identical regardless of which head(s) the plan selected — every head consumes the same point cloud. What varies is which *additional* target tensors you build: only build what the plan's selected head(s) actually need, and omit the rest entirely rather than filling them with placeholder data. This isn't cosmetic — EveNet's preprocessing code (`preprocessing/sanity_checks.py`, `preprocessing/helper.py`) treats each task's tensors as present-or-absent: a task with no tensors in the npz is simply skipped (logged as "inactive"), with no shape/dtype requirement and no attempt to log1p-transform something that isn't there. Inventing a fake shape for an unused task adds risk for no benefit.

Templates referenced below live in `.claude/agent-resources/data-converter/`.

## Step 1: Create directory structure

**`<run_dir>` (used throughout this pipeline, by every subagent) is `<evenet_full>/run/<project_name>/` — never bare `<evenet_full>/run/`.** This matters: without a project-specific subdirectory, a second analysis run in the same environment would silently overwrite this one's converted data, checkpoints, and predictions. You're the first agent to run after plan approval, so you're the one who creates this whole tree — don't create only the pieces your own phase needs and assume later agents will create theirs; `sbatch` in particular will fail at submission if `<run_dir>/logs/` doesn't already exist when `fine-tuner` submits its job.

```bash
mkdir -p <run_dir>/data_processed/npz
mkdir -p <run_dir>/data_processed/parquet
mkdir -p <run_dir>/ckpts        # (or ckpts_0, ckpts_1 for 2-fold)
mkdir -p <run_dir>/predict      # (or predict_0, predict_1 for 2-fold)
mkdir -p <run_dir>/logs
mkdir -p <run_dir>/output
```

## Step 2: Generate the event_info YAML

Copy `.claude/agent-resources/data-converter/event_info_template.yaml` to `EveNet-Full/share/event_info/<project_name>_process.yaml`, filling in per the plan:

- `INPUTS.GLOBAL.Conditions`: **leave this section exactly as the template has it** (the fixed 10-feature list matching the pretraining corpus's own schema — `met`, `met_phi`, `nLepton`, `nbJet`, `nJet`, `HT`, `HT_lep`, `M_all`, `M_leps`, `M_bjets`) **even if the plan has no real global conditions.** Don't leave it `{}` — an empty `Conditions:` produces a zero-length feature list, and `evenet/control/event_info.py` builds `torch.tensor([])` from it, which PyTorch defaults to `float32` instead of `bool` — this crashes `torch.where` in `Normalizer.__init__` (`evenet/network/body/normalizer.py`) at model-construction time. Confirmed by hitting exactly this in production: training completed, but prediction crashed with `RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Float`, traced to an empty `Conditions: {}`. Beyond just avoiding the crash, matching this exact schema is also what lets the pretrained checkpoint's `GlobalEmbedding` weights actually load (`safe_load_state` matches by shape; a different feature count means those weights get silently skipped instead of reused). If the plan's data has values for any of these 10 (or a genuinely custom condition not on this list), see Step 3 item 6 for how to populate them — the YAML declaration itself doesn't change either way.
- `GENERATIONS.Neutrinos`: the plan's invisible-particle features/normalization **only if `TruthGeneration` is selected**; otherwise leave `{}`
- `GENERATIONS.GlobalTargets`: the plan's chosen condition(s) **only if `GlobalGeneration` is selected**; otherwise leave `{}`
- `CLASSIFICATIONS`/`CLASSLABEL`: if `Classification` is selected as a real trained head, list the plan's actual category names under `CLASSLABEL.EVENT.signal`; otherwise leave the single generic project-name placeholder (harmless — Classification stays `include: false` in the finetune YAML either way)
- `EVENT:`/`PERMUTATIONS:`: **only if `Assignment` is selected**, fill these from the plan's resonance topology, following the pattern used in `EveNet-Full/share/event_info/multi_process.yaml` and `pretrain.yaml` (read one of those first as a concrete reference — don't invent the schema from scratch):
  ```yaml
  EVENT:
    <process_name>:
      <resonance_name>:
        - <daughter1>
        - <daughter2>

  PERMUTATIONS:
    <process_name>:
      <resonance_name>:
        - [ <daughter1>, <daughter2> ]   # only for daughters the plan marked symmetric
  ```
  For most plans this is a single `<process_name>` (one fixed topology). If `Assignment` isn't selected, leave both empty as before — this schema is only needed for `Assignment`/`Segmentation`/`Regression`, and only `Assignment` is currently supported.

## Step 3: Generate the to-npz conversion script

Write `<run_dir>/data_processed/npz/to_npz.py`, implementing the plan's slot mapping exactly — including any identification/ordering rule `physics-planner` verified empirically (e.g. mass-hypothesis-based particle ID rather than fixed array index). Everything from step 2 onward below is the same regardless of input format — only *how you read a per-event field* differs. Name the script `to_npz.py` regardless of format (not `root_to_npz.py`) since `data-reconverter` imports from it later and needs one stable name to import from either way.

**If the plan's input format is ROOT**: use `uproot`, exactly as before. Accept `--input_dir`, `--output_dir`, `--tree_name` as CLI arguments; read all `.root` files in `input_dir`.

**If the plan's input format is `.pt`**: use `torch.load(path, map_location='cpu', weights_only=False)` against the field names/structure `physics-planner` actually found (don't assume it matches the ROOT convention). Accept `--input_dir`, `--output_dir` as CLI arguments — same as ROOT, process every `.pt` file found in the directory (if the user's data is a single `.pt` file, treat `--input_dir` as that file's parent directory, matching the same "process everything found here" convention as ROOT rather than introducing a separate single-file argument the shell templates don't actually pass). No `--tree_name` (that's a ROOT-only concept).

**Structure it so `data-reconverter` can import and reuse the read/slot-mapping logic later** — put the per-file "read input, build `x`/`x_mask` (and the other per-event tensors)" logic in a standalone function (e.g. `process_file(path, ...)`), separate from the CLI/split/save code in `if __name__ == "__main__":`. `data-reconverter` needs to call exactly this function directly (not reimplement it) to rebuild the feature array for event-matching later — if it's all written inline under `__main__` with no importable function, that import will simply fail.

The script must, for either format:

1. Read all input files (see above)
2. Build `x` tensor `(N, 18, 7)` per the plan's slot mapping — compute `E = sqrt((pT·cosh(eta))² + m²)` where energy isn't a direct field; pad unused slots with zeros
3. Build `x_mask` `(N, 18)` — `True` for valid slots
4. **Only if `TruthGeneration` is among the plan's selected heads**, build invisible particle tensors from the plan's target definition: `x_invisible` `(N, N_nu, F_nu)`, `x_invisible_mask` `(N, N_nu)`, `num_invisible_raw`/`num_invisible_valid` `(N,)` int32. **Store raw physical values — do not manually apply `log1p` here.** The downstream `preprocess.py` step applies `log1p` automatically to any feature declared `log_normalize` in the event_info YAML (for both `x` and `x_invisible`); applying it again here double-transforms the target. Verify this against the actual `preprocess.py` log output in Step 5 before trusting it silently. **If `TruthGeneration` is not selected, don't add these keys to the npz dict at all** — not even zero-filled placeholders.
5. **Only if `Assignment` is among the plan's selected heads**, build the resonance-assignment tensors: `assignments-indices` `(N, R, D)` int — slot index per daughter, per resonance (a `torch.gather`-style index into the 18-particle dimension, confirmed from `evenet/network/metrics/assignment.py`); **pad missing/invalid daughters with `-1`, never `0`** — `0` is a real, valid slot index, so padding with it would silently misassign slot 0 as a daughter rather than mark the slot absent (this exact convention is documented on the EveNet docs site, not just inferred). `assignments-mask` `(N, R)` bool — whether each resonance is fully present in the event. `assignments-indices-mask` `(N, R, D)` bool — per-daughter validity (`False` exactly where `assignments-indices` is `-1`). `subprocess_id` `(N,)` int — `0` for every event in the common single-topology case. `process_names` `(N,)` string — matching process name per event.

   **`process_names` must only be included in the train split's npz — never val's or test's.** Found the hard way on a real analysis: `preprocessing/helper.py`'s npz-to-parquet step reads `process_names` to compute per-process statistics feeding `normalization.pt` (train-only, by design — statistics must never be fit on val/test), and as an incidental side effect of that same code path, *only* the train split gets this column stripped back out before writing to parquet. Val/test have no equivalent cleanup. So if your script writes `process_names` uniformly across all three splits (the natural default, since the rest of your split-writing logic is otherwise identical across splits), it silently survives into `val.parquet`/`test.parquet` — where it later crashes Ray's `iter_torch_batches` collate during standalone prediction (`predict.py` fails after 0 iterations, opaque error), since Ray's default collate only supports numeric/bool dtypes and `process_names` is a string. When writing each split's npz in Step 7, include `process_names` only for `train`; omit it entirely from `val`/`test`. Confirmed safe to omit there: every reference to `process_names` in the codebase resolves to `event_info.process_names` (the config-level list from the YAML), never a per-event data field read out of a batch — and the reference dataset published by the EveNet team itself doesn't write this column in *any* split, corroborating it isn't needed once past the train-side statistics computation.

   **This is newer, less-verified territory than the TruthGeneration path** — before trusting your own construction, read `evenet/control/event_info.py`'s `EventInfo.__init__` (and what it does with `event_particles`/`product_particles`/`permutations`) to confirm the exact shape and ordering convention your script needs to match, the same way you'd verify the log1p behavior in Step 5 rather than assume it. If something about the expected shape genuinely doesn't resolve after reading that code, report the ambiguity rather than shipping a guess. **If `Assignment` is not selected, don't add these keys to the npz dict at all.**

   **Matching daughters by geometric ($\Delta R$) proximity — two things to get right, found the hard way on a real analysis:**
   - **Exclusive/joint matching across resonances for same-type daughters.** If more than one resonance has a same-type daughter matched independently by nearest-$\Delta R$ (e.g. a $b$-jet daughter on each of two different resonances, both matched against the jet collection separately), nothing stops both from resolving to the *same* slot when their two truth targets happen to be geometrically close — a real, if not always common, occurrence. Don't match each daughter's truth target independently; compute $\Delta R$ from every such truth target to every candidate slot jointly, and if two truth targets nearest-match the same slot, assign it to whichever is actually closer and let the other's search exclude that slot and fall back to its next-best remaining candidate (still subject to the cutoff).
   - **`assignments-mask` should prefer truth information over pure geometric proximity when the plan reports it's available.** If the plan's resonance topology section states a verified truth-flavor field (see `physics-planner.md`) for a jet-type daughter — report its exact name and the value identifying the correct particle type, e.g. `partonFlavour == 5` for a $b$-jet; **the field name varies by dataset, never hardcode one you've seen before, always use exactly what the plan states** — then `assignments-mask` for that daughter should require *both* the existing $\Delta R$ match *and* the matched slot's flavor field equaling the plan-specified value, not $\Delta R$ alone. `assignments-indices` itself is unaffected (still the pure nearest-$\Delta R<0.4$ slot, matching what the plan describes) — only the validity flag gets the extra truth gate, since a geometrically-nearby wrong-flavor jet can pass a pure distance cut but shouldn't count as a valid training target. If the plan states no reliable truth-flavor field exists for a daughter, fall back to $\Delta R$-only for both the index and the mask, as before.
6. Build required metadata (always, regardless of head): `num_vectors`/`num_sequential_vectors` `(N,)` (from `x_mask.sum(axis=1)`), `event_weight` `(N,)` float32 ones. `conditions` `(N, 10)` float32 — **always 10 columns, matching the fixed `INPUTS.GLOBAL.Conditions` schema from Step 2 in the same order** (`met, met_phi, nLepton, nbJet, nJet, HT, HT_lep, M_all, M_leps, M_bjets`): populate whichever of these the plan's data actually has with real values, zero-fill the rest — never fewer than 10 columns, regardless of how many the plan's analysis actually uses, since the column count must match what Step 2's YAML declares or model construction will fail on a dimension mismatch. `conditions_mask` `(N, 1)` bool — this is a single per-event flag (not one per condition column) for whether global conditions are meaningfully populated for that event; `True` if the plan has any real conditions, `False` if all 10 are zero-filled placeholders. For `classification` `(N,)` **int32** (the documented input dtype — the framework casts it up to int64 internally, matching what we saw in our own successful run): if `Classification` is selected as a real trained head, build it from the plan's category rule; otherwise use zeros (a harmless placeholder — it's only ever read if `Components.Classification.include: true`, which `fine-tuner` only sets when the plan actually selected it).
7. Split per the plan's split mode:
   - **standard**: shuffle with a fixed seed, split per the plan's ratio (default 80:10:10), save `train.npz`/`val.npz`/`test.npz`
   - **2fold**: even-index events → `train.npz`, odd-index events → `test.npz`, no `val.npz`

## Step 4: Generate the preprocessing script

Copy `.claude/agent-resources/data-converter/preprocess.sh` to `<run_dir>/preprocess.sh`, replacing `<project_name>`. For 2-fold mode, use the train/test-only variant (no `--val`, no `val` in the symlink loop) — the template has both variants marked.

## Step 5: Run data preparation (inside container, no GPU)

Copy `data_prep_nersc.sh` or `data_prep_docker.sh` from `.claude/agent-resources/data-converter/`, fill placeholders (`<evenet_full>`, `<container_image>`, `<input_data_dir>`, `<project_name>` — the templates `cd` into `run/<project_name>/` rather than bare `run/`). **`<tree_name>` only applies if the plan's input format is ROOT** — the templates pass it as a `--tree_name` flag conditionally; if the input is `.pt`, drop that flag from the `to_npz.py` invocation entirely rather than passing an empty/placeholder value. Run it. **Read the preprocessing log output carefully** — it states exactly which features get `log1p` applied automatically (look for `"Applying np.log1p to log-scale features"`); cross-check this against what Step 3's script already did, and fix+rerun if you find a mismatch (double-application or missing application) rather than assuming it's correct.

Verify outputs exist:
- Standard: `data_processed/npz/{train,val,test}.npz`
- 2-fold: `data_processed/npz/{train,test}.npz`
- Both: `data_processed/parquet/{train,test}[,val]/`, `normalization.pt`

## Output

Report back to the orchestrator (for hand-off to `fine-tuner`):
- Event counts per split
- Path to `data_processed/parquet/`
- Path to `normalization.pt`
- Any log1p/normalization mismatch you found and how you resolved it
