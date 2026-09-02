# Unfolding (RooUnfold)

Read this whenever the approved plan's downstream observable involves unfolding a fiducial/reco-level distribution to a truth-level one (e.g. `RooUnfoldSVD`/`RooUnfoldBayes` against a truth region defined by a mass window or similar).

## 1. Build the response matrix with `Fill` and `Miss`

Build the response over the *full* truth-eligible population, not just events that pass both the fiducial cut and the truth-region cut. For every event with a defined truth value:

```python
if not isnan(truth_val) and not isnan(reco_val):
    response.Fill(reco_val, truth_val, weight)
elif not isnan(truth_val):
    response.Miss(truth_val, weight)
# an event with a valid reco value but no truth counterpart is a response.Fake(reco_val, weight)
# -- check whether that case can occur in your analysis before assuming it can't.
```

Use `isnan()`/`math.isnan()` to test validity — never `!= float('nan')` or `!= np.nan`. Those are **always** `True` in Python regardless of the operand (`float('nan') != float('nan')` is `True`), which silently makes any `Miss()`/`Fake()` branch gated on them unreachable.

**Never restrict response construction to only the events passing both the fiducial cut and the truth-region cut.** Every truth-valid event must contribute — as `Fill` if it also has a reco value, as `Miss` if it doesn't. `Miss` is what carries the efficiency/acceptance correction; without it the response is both statistically starved and missing the information RooUnfold needs to correct for acceptance.

**Bin truth and reco/measured jointly across the full available range in one unfold, then slice to the target window afterward** — don't pre-cut to the target window before unfolding runs. Bin the observable together with whatever variable defines the truth region (e.g. the observable in its own bins × a handful of wide windows of that variable, spanning the *whole* sample), run one unfold over that joint binning, and only read out the slice matching the plan's actual target window (e.g. `m_tt<400 GeV`) afterward. This keeps the response well-conditioned across the full available statistics.

**Verify before trusting the result**: every unfolded value is within its observable's physical bound, no bin has negative content, and a closure check passes (§3.1) — plot unfolded against truth per bin for every variable, not just the diagonal/headline ones.

## 2. Truth-side quantities must come from real truth particles

The truth side of every `Fill`/`Miss` (and any closure check) needs genuine truth-level 4-vectors — never the model's predicted objects, and never reconstructed objects matched to truth via `assignments-mask`/`assignments-indices` (a "match" is a reco-level quantity, subject to its own matching inefficiency, not a clean truth reference).

Get real truth particles into the pipeline the same way any other pass-through field gets there: name them in the plan's `extra_save` list, have `data-converter` write them as ordinary pass-through parquet columns, let `fine-tuner`/`predictor` carry them unchanged into the prediction `.pt`.

**One field per particle, not one field per scalar component.** Pack each truth particle's full kinematics (`[pt, eta, phi, mass]`, plus e.g. `pdgId` if the particle's identity/charge isn't already fixed by the topology) into a single array-valued `extra_save` field per particle, rather than separate `<name>_pt`/`<name>_eta`/`<name>_phi`/`<name>_mass` fields. Array-valued columns round-trip through parquet as fixed-width numeric columns and survive Ray's `iter_torch_batches` collate cleanly — a different, safer risk class than a string-typed column (which can crash that same collate step).

**Prefer an already-defined truth-particle source over deriving one yourself, whenever one exists.** If the plan's input format is already `.pt` and the input file already carries these truth particles under its own field names, reuse those names/values directly rather than re-deriving the same particles independently (e.g. via your own mother-chain/matching logic against a lower-level generator record). If you do have to derive truth particles yourself, and an independently-produced reference sample for the same underlying data exists, cross-check your derived particles' kinematics against it (e.g. compare pt/eta/phi distributions particle-by-particle) before trusting the result — an independently-derived truth extraction can silently diverge from a reference one even when both run without error.

**State the truth-particle list explicitly in the plan.** `physics-planner`'s `extra_save` field list (its own Step 16) must name every truth particle the response matrix's `Fill`/`Miss` truth side will need, and everything a closure check will want to plot against, decided up front.

## 3. Verification plots to produce

1. **Unfolded vs. truth, per bin, for every variable being unfolded** — not just the diagonal/headline ones. Overlay unfolded (points) against truth (line) per variable so agreement is visible bin-by-bin, not only in an aggregated mean.

2. **Mean-and-uncertainty summary, truth vs. unfolded, across all variables in one compact plot** — one point (or bar) per variable for truth and for unfolded, each with its uncertainty as an error bar. This is a different check from #1: #1 shows whether the *shape* matches bin-by-bin, this shows whether the *summary number* matches at a glance across every variable at once — a shape mismatch that happens to average out won't show here, and a small systematic mean shift that's hard to spot in a busy shape overlay will.

3. **If the plan defines a downstream scalar observable from the unfolded distribution** (e.g. `D = |C_11+C_22|-C_33` from three unfolded matrix elements), report it three ways side by side, not the unfolded value alone: the **truth value** (direct from truth particles, no reco or unfolding involved), the **raw reco value** (fiducial-level, before any unfolding), and the **unfolded value** — each with its own event count and uncertainty. A correct unfold should land close to the truth-direct value; seeing all three together makes a bad unfold visible immediately rather than something that has to already be suspected before it's checked.

## 4. Statistical uncertainty — analytic, from RooUnfold's own covariance, not toys

If the plan calls for stat-only uncertainty via analytic error propagation (not toy/bootstrap resampling):

- **Read out the unfolded histogram with `Hunfold(2)`.** RooUnfold's `ErrorTreatment` enum: `kNoError=0` (no errors, just `sqrt(N)`), `kErrors=1` (diagonal elements only of the analytic covariance), `kCovariance=2` (the full analytic covariance matrix, propagated through the unfolding's matrix inversion), `kErrorsToys=3`/`kCovToys=4` (toy-throwing based). Use `2` for stat-only analytic propagation; don't reach for `kErrorsToys`/`kCovToys` unless the plan specifically asks for toy uncertainties.

- **Propagating per-bin errors to a derived mean-based quantity** (e.g. turning an unfolded 1D histogram into `C_ij = -9<cosθ_i cosθ_j>`, a weighted mean over bins):

  ```python
  mean = sum(bin_centers * bin_counts) / total_count
  variance = sum(((bin_centers - mean) / total_count)**2 * bin_errors**2)
  uncertainty = sqrt(variance)
  ```

- **Propagating from several such quantities to a further-derived observable** (e.g. `D = |C_11+C_22|-C_33` from three independently-unfolded matrix elements): quadrature sum of the partial derivatives —

  ```python
  D_err = sqrt(C11_err**2 + C22_err**2 + C33_err**2)   # here every partial derivative has magnitude 1
  ```

  This assumes negligible correlation between the quantities going into it — true if each was unfolded as its own independent 1D histogram. If they were unfolded jointly (a genuine multi-dimensional unfold with real cross-covariance), use the full covariance matrix instead of quadrature-summing independent errors, and say explicitly which of the two applies.
