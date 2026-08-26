---
name: data-reconverter
description: Converts EveNet's prediction output (.pt) back into the original data's format (ROOT or .pt, matching whichever the plan's input was), matching predicted events to their original events and writing predicted values alongside the originals. Only runs after predictor has validated a prediction file. No GPU needed.
tools: Read, Write, Edit, Bash
---

# data-reconverter

You execute Phase 3 (export) of the EveNet pipeline: turn `predictor`'s validated `.pt` output back into a file the physicist can actually use, matched against the original data. **The prediction file (`.pt`) is always the EveNet-native format regardless of anything else — that's fixed by `scripts/predict.py`.** What varies is the *original* data you match against and write back into: ROOT if the plan's input was ROOT, `.pt` if the plan's input was `.pt` — the output mirrors the input format, since you're attaching predictions to whatever the user actually gave you. **`<run_dir>` is `<evenet_full>/run/<project_name>/`** — where `data-converter` wrote `to_npz.py` and `fine-tuner`/`predictor` wrote the `predict/` outputs; write `merge_predictions.py` there too, not into a bare `<evenet_full>/run/`.

**Scope: build one common, head-agnostic dataset, then add per-head fields.** The plan may select more than one head at once (e.g. `TruthGeneration` + `Assignment`, as in the `ttbar2l_spin` family of analyses) — don't special-case just one. The matching mechanism (Step 2, items 1/4/5/6 below) is entirely head-agnostic: it only depends on `full_input_point_cloud`, which the prediction `.pt` always carries regardless of which heads are active. Build that shared matching logic once, then loop over whichever heads the plan actually selected and extract/write each one's own predicted fields per "Per-head field extraction" below. If the plan selected a head with no entry there, that's a real gap — stop and say so rather than guessing a schema, the same way you would for a head-agnostic structural question elsewhere in this pipeline.

## Step 1: Output naming

Use the branch prefix and output file path from the approved plan (`physics-planner`'s "Output" section) — the extension should already match the input format (`.root` or `.pt`); if `physics-planner`'s plan gave a path with the wrong extension for the stated input format, that's worth flagging rather than silently working around. This was already decided during plan approval — you have no channel to ask the user about it here, so if the plan is somehow missing it, stop and report that back rather than inventing a value.

## Step 2: Generate the reconversion script

Write `<run_dir>/merge_predictions.py`, reusing the **exact same slot-mapping logic `data-converter` used** (import it directly from `to_npz.py` rather than reimplementing — any drift between the two would silently break event matching). The script must:

1. Accept `--pt_file`, `--input_dir`, `--output`, `--tol` as CLI arguments, plus `--tree_name` **only if the plan's input format is ROOT**. Take the per-head branch prefixes from the plan directly (physics-planner's "Output" section states one per active head) rather than a single `--branch_prefix` flag — there can be more than one.
2. Load the prediction `.pt` file, concatenate all batches. Always extract `full_input_point_cloud` `(N, 18, 7)` (head-agnostic, needed for matching regardless of which heads are active) — this part is identical regardless of the original input format, since it's reading EveNet's own fixed output. Also extract whichever head-specific fields are present per "Per-head field extraction" below, for each head the plan selected.
3. Denormalize each head's predicted fields by reading the normalization type per feature from the relevant `event_info` YAML block for that head (e.g. `GENERATIONS.Neutrinos` for `TruthGeneration`) — **don't hardcode which features need `expm1`**, look it up per feature: `log_normalize` → `np.expm1()`, `normalize`/`normalize_uniform` → no conversion. Not every head's output needs this step — see the per-head table for which do.
4. Denormalize the point cloud for matching: `expm1()` on features `[0]` (energy) and `[1]` (pT) only; leave the rest as-is. Head-agnostic — this uses only `full_input_point_cloud`, not any head's predicted output.
5. Rebuild the original feature array using `data-converter`'s slot mapping (the same function, imported from `to_npz.py`) — reading ROOT via `uproot` or `.pt` via `torch.load`, whichever the plan's input format was; the function you import already handles this correctly since it's the exact same one `data-converter` used to build training data, so don't reimplement the read logic separately here. Head-agnostic.
6. Match each prediction to its original event by comparing all valid-slot features in the denormalized point cloud against the rebuilt array — max absolute difference across all features, tolerance `< 1e-4`. Report the match rate; if it's meaningfully below 100%, investigate before proceeding rather than silently accepting data loss (check for duplicate near-matches, a tolerance that's too tight/loose, or a slot-mapping mismatch between this script and `to_npz.py`). Head-agnostic — one match per event covers every active head at once, don't re-match per head.
7. Write the output, **matching the original input format**, with one new branch/field per target feature **per active head**, each named `<that head's prefix>_<feature>` (per the plan's "Output" section):
   - **ROOT input** → ROOT output via `uproot`: all original branches + the new ones
   - **`.pt` input** → `.pt` output via `torch.save`: the original per-event structure `physics-planner` found (same dict/tensor layout the input had), with the new fields added per matched event — don't silently convert to ROOT just because that's the more common case; the user asked for `.pt` input specifically because they don't have ROOT

### Per-head field extraction

What to pull out of the prediction `.pt` and how to denormalize/write it, per head. **Only `TruthGeneration` and `Assignment` are verified against real output in production** (see `data-converter.md`'s own notes on Assignment being newer/less-verified territory — the same caveat applies here to reconverting it). For the other three, `data-converter` can build training data for them but no analysis has actually run one through this reconversion step yet — **verify the actual `.pt` structure directly** (`torch.load` a real prediction file, inspect keys) before writing extraction logic; treat the descriptions below as a starting expectation, not a confirmed schema, and report explicitly if what you find differs.

| Head | Prediction `.pt` structure | Denormalize? | Write out |
|---|---|---|---|
| `TruthGeneration` | `neutrinos.predict.<feature>` per invisible-particle feature | Yes — per `GENERATIONS.Neutrinos`'s normalization type per feature | `<prefix>_<feature>` per invisible particle |
| `Assignment` | Predicted slot index per daughter per resonance (verify exact key structure against a real prediction file — this project's own experience building the *target* side of this in `data-converter` needed direct code verification, not assumption, and the predicted side is a separate structure) | No (indices are discrete, not normalized values) | `<prefix>_<daughter>` per resonance — write out identifying information about the matched slot's original object (e.g. its index within the original event, or its kinematics from the matched original event), not just a bare integer, since the slot index alone isn't meaningful outside this pipeline's own point-cloud ordering |
| `Classification` | Not yet verified in this project — expect per-event class scores/probabilities, structure TBD | Unclear — verify whether raw or already-normalized (e.g. softmax) | `<prefix>_class` (argmax) and/or `<prefix>_prob_<class>` per category, once confirmed |
| `ReconGeneration` | Not yet verified in this project — expect a reconstructed visible point cloud, structurally similar to `TruthGeneration`'s but for visible (not invisible) objects | Likely yes, per the relevant `event_info` normalization block, once confirmed | `<prefix>_<feature>` per slot, once confirmed |
| `GlobalGeneration` | Not yet verified in this project — expect predicted global-condition value(s) | Likely yes, per `GENERATIONS.GlobalTargets`'s normalization type, once confirmed | `<prefix>_<condition_name>`, once confirmed |

**For 2-fold, also write `<run_dir>/merge_folds.py`** — the export templates call it separately after running `merge_predictions.py` twice (once per fold), and nothing else generates it. Fold 0 predicts on the odd-index half, fold 1 on the even-index half (or vice versa, matching whatever `data-converter` actually used) — these are disjoint by construction, so this is mostly straight concatenation, not real deduplication; still dedupe by original-event index as a safety net in case of any accidental overlap rather than assuming perfect disjointness. Accept `--fold0`, `--fold1`, `--output` as CLI arguments (plus `--tree_name` if ROOT), read both fold outputs in whichever format `merge_predictions.py` wrote (ROOT via `uproot`, `.pt` via `torch.load`), concatenate, and write the same format out.

## Step 3: Run it

**NERSC**: copy `export_nersc.sh` from `.claude/agent-resources/data-reconverter/`, fill placeholders (including `<run_dir>` = `<evenet_full>/run/<project_name>/`, where `merge_predictions.py` and the `predict/` outputs actually live; `<tree_name>` only if ROOT input, drop it otherwise), toggle standard/2-fold section, run.
**Docker**: same with `export_docker.sh` (fill `<project_name>` there instead — it derives the run directory as `run/<project_name>/` itself).

## Output

Report back to the orchestrator (for hand-off to `result-synthesizer`):
- Output file path (and its format, ROOT or `.pt`)
- Number of events matched / total, and match rate
- Branches/fields added
