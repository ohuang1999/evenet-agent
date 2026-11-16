# MadAgents request template

`mc-generator` renders this into `<run_dir>/mc_generation/request.md` and sends it as the **first** message of
the MadAgents session (`madagents_turn.sh start ... request.md`). Every `<...>` is filled from the approved
plan's **MC request** section; nothing is invented. A field the plan leaves open stays open in the request
and is listed under "Open in the plan" so MadAgents asks instead of choosing.

Keep the protocol block verbatim — `mc-generator` parses the `@@` markers.

---

```markdown
# MC production request: <project_name>

## How this conversation works (read first)

You are being driven **headless by an automated client** (`mc-generator`, part of an EveNet analysis
pipeline), not by a person at a terminal. Nobody can answer an interactive prompt. Rules:

1. **Never guess a physics or cost decision.** If anything below is missing, ambiguous, or would need to
   change (a card value, a PDF, a scale, matching, a generator cut, the event target, the budget), end your
   turn with a `@@QUESTIONS` block. Purely technical choices (directory layout, retrying a crashed split,
   scratch vs CFS) are yours.
2. **Do not wait on Slurm inside a turn.** After you submit jobs, end the turn with a `@@JOBS` block. You
   will be resumed when the jobs have left the queue, with their `sacct` states.
3. Run a **cards-only dry run** per process and compare it with the reference run (if one is given)
   **before** any production submission; report the comparison in the turn that submits.
4. **Where files go.** Every script you write for this production — `.mg5` scripts, launch/card files,
   sbatch scripts, task lists, counting/selection and check scripts — goes in `<run_dir>/mc_generation/scripts/`,
   never in your own folder (it is shared across runs). Process/work directories go in `$SCRATCH` (purged after
   ~8 weeks idle; treat it as scratch only). Deliverables go in `<mc_output_dir>`. Before ending with `@@DONE`,
   copy each run's logs (MG, Pythia8, Delphes, MadSpin) next to its ROOT file in `<mc_output_dir>`. Never
   modify the reference samples.
5. End **every** turn with exactly one of these blocks, as the last thing in your reply:

   @@QUESTIONS
   1. <question> — options: <a / b / ...> — why it matters: <physics or cost impact>
   @@END

   @@JOBS
   jobids: <id> <id> ...
   what: <one line per job or array: process, runs, events, expected wall time>
   node_hours_requested: <number>
   @@END

   @@DONE
   index: <mc_output_dir>/samples.csv
   summary: <path to your summary markdown>
   @@END

   @@BLOCKED
   reason: <what stops you, e.g. a refused permission, a missing tool, a failed check you cannot fix>
   @@END

## What to produce

Processes (one row each):

| process | model | generate / decay line | events target | target counted after |
|---|---|---|---|---|
<one row per process from the plan>

- Collider / energy: <e.g. pp, 13 TeV>
- Parameters: <param_card values or "model defaults"; benchmark points if scanning>
- Generator-level cuts: <run_card cuts, or "MG defaults">
- PDF: <set / LHAPDF id>   Scales: <dynamical_scale_choice etc.>
- Matching / merging: <e.g. MLM, xqcut=..., or "none">
- Decays: <MadSpin card content, or "in the generate line">
- Shower / hadronisation: <Pythia8 card or defaults>
- Detector: Delphes card <path> (md5 <md5>) — must be used unchanged
- Seeds: <policy, e.g. unique per run, Pythia seed = MG seed>
- Reference run for settings: <path or "none"> — every card value not stated above follows this run

## Event targets and the selection used to count them

<"events target" is counted after this selection, per process, over train+test.
Give the exact selection: objects, cuts, and which Delphes branches. If "Delphes events", say so.>

## Resources

- NERSC account: <account>; QOS/constraint: <e.g. regular, cpu>
- Node-hour budget for the whole production: <N>; storage budget: <TB>
- Ask before exceeding either.

## Deliverables in <mc_output_dir>

1. Delphes ROOT files, one directory per process and run, with that run's banner, `Cards/` and logs.
2. `samples.csv` following the schema in the agent's samples-index spec (columns listed below).
3. A summary markdown: what was run, card checks (dry-run diff, after-run diff, Delphes md5), the cross
   section convention per process, event counts before and after the selection, node-hours used, and
   anything unusual (warnings, retried splits).

`samples.csv` columns: <copied from samples-index.md>

## Open in the plan

<list of fields the plan left open — ask about these with @@QUESTIONS before submitting production; or "none">
```
