# MC samples index (`samples.csv`)

The one hand-off from `mc-generator` to the rest of the pipeline. `physics-planner` (second pass) reads it
for the **Event weights** section and the input-data location; `data-converter` takes cross sections and
event totals from it. One row per generated run (a process usually has several runs with different seeds).

| column | meaning |
|---|---|
| `process` | process label from the plan's MC request (e.g. `WZ_QCD`, `GM_H5p_m375`) |
| `category` | `signal` or a background group name from the plan |
| `run_dir` | absolute path of the run directory |
| `root_file` | absolute path of the Delphes ROOT file |
| `seed` | MG seed (and Pythia seed, if the plan's policy ties them) |
| `n_lhe` | events in the LHE file |
| `n_delphes` | entries in the Delphes tree (after matching for MLM samples) |
| `sigma_pb` | cross section to normalise this run, in **pb** |
| `sigma_convention` | `lhe` (plain), `matched` (MLM: matched sigma, or sigma_lhe × n_delphes / n_lhe if the banner lacks it), or `sigma_x_br` (MadSpin: decayed-banner sum) |
| `n_selected` | events passing the plan's counting selection (the one the events target refers to) |
| `delphes_card_md5` | md5 of the Delphes card the run used |
| `card_check` | `identical`, or `seed_nevents_only`, or a short note of any other difference vs the reference run |
| `superseded` | `true` for runs that must not be used (kept for the record), else `false` |
| `notes` | anything unusual (retried Delphes splits, warnings) |

Rules:
- **Per-event weight** downstream = `sigma_pb × 1000 × L[fb⁻¹] / N`, where N is the number of events of
  that process (or split) **before any selection** — `n_delphes` summed over the runs used. Using
  `n_selected` as N is wrong: it double-counts the selection efficiency.
- Every number in the file is read from the run's own outputs (banner, Delphes tree, card files), never
  copied from the plan or from MadAgents' prose.
- CRLF/LF: write LF.
