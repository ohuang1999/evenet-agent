"""
Generic contiguous multi-type slot packing for EveNet's (N, 18, 7) point cloud.

Implements the packing convention from physics-planner.md Step 4: particles are
packed contiguously in a stated type-priority order (e.g. jets, then leptons),
NOT into fixed per-type slot ranges -- which type occupies a given slot index
varies event-to-event depending on that event's own per-type multiplicity that
event (e.g. an event with 3 jets has its first lepton at slot 3; an event with
7 jets has its first lepton at slot 7). Trailing slots left over after every
real particle is placed are zero-filled with mask=False. isLepton/btag/charge/
etc. at a given slot always come from whichever real particle got packed
there -- never a constant tied to the slot's position.

Truncation policy (when an event's total real-particle count across all types
exceeds n_slots): types are truncated in a stated priority order too, and this
order does NOT have to match the packing order. Concretely, for a jets+leptons
dilepton analysis: pack jets-then-leptons in the array, but truncate jets
first on overflow -- the analysis's own selection is defined by its leptons,
so silently dropping a lepton is a much bigger correctness problem than
dropping one extra low-pT jet. physics-planner's plan must state both orders
explicitly; don't assume they're the same order just because that's simpler.

Two named reference implementations sit alongside this file for inspiration --
adapt the *pattern*, not the literal field names, since those always vary by
dataset (see data-converter.md's own note on this):
  - reference/root_example_softgamma.py   -- ROOT, fixed single-particle-per-slot
                                              decay topology (no variable
                                              multiplicity / no truncation
                                              question -- a different case from
                                              the one this helper solves, shown
                                              for the general shape of a
                                              converter script, not the packing
                                              algorithm itself)
  - reference/h5_example_nu2flow_convertor.py -- h5, contiguous packing of a
                                              fixed-count type (exactly 2
                                              leptons) followed by a
                                              variable-count type (jets) --
                                              closer to this helper's own
                                              case, but still doesn't need a
                                              truncation policy since it never
                                              caps the jet block independently
"""
from __future__ import annotations

import numpy as np


def pack_slots(
        type_features: list[np.ndarray],
        type_counts: list[np.ndarray],
        pack_order: list[int],
        truncate_order: list[int],
        n_slots: int = 18,
        n_features: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    type_features : one (N, k_max_t, n_features) float array per particle type,
        already sorted *within* that type per event (e.g. pT-descending) and
        already zero-padded out to that type's own per-event max multiplicity
        k_max_t (k_max_t may differ across types and does not need to relate
        to n_slots at all -- e.g. a type with real per-event counts up to 14
        is fine even though n_slots=18 total across every type).
    type_counts : one (N,) int array per type -- the real (non-padding) count
        of that type in each event. Must satisfy
        type_counts[t][i] <= type_features[t].shape[1] for every event i.
    pack_order : indices into type_features giving the slot-packing order
        (e.g. [0, 1] means type 0's surviving particles go first, then
        type 1's, contiguously).
    truncate_order : indices giving truncation priority when an event's
        combined real count exceeds n_slots -- the type listed first is
        truncated first (hardest). Does not need to match pack_order.
    n_slots, n_features : output tensor dimensions (18, 7 for EveNet's
        pretrained point-cloud schema).

    Returns
    -------
    x : (N, n_slots, n_features) float32
    x_mask : (N, n_slots) bool -- True for every slot a real particle landed in
    """
    n_types = len(type_features)
    N = type_features[0].shape[0]
    assert len(type_counts) == n_types == len(pack_order) == len(truncate_order), \
        "type_features/type_counts/pack_order/truncate_order must all cover the same set of types"
    assert sorted(pack_order) == sorted(truncate_order) == list(range(n_types)), \
        "pack_order and truncate_order must each be a permutation of range(n_types)"

    counts = np.stack(type_counts, axis=1).astype(np.int64)  # (N, n_types)
    for t in range(n_types):
        assert np.all(counts[:, t] <= type_features[t].shape[1]), \
            f"type {t}: type_counts exceeds type_features' own padded width"

    # --- Step 1: truncate, working through truncate_order until every event fits ---
    keep = counts.copy()
    running_total = counts.sum(axis=1)  # (N,)
    over = running_total > n_slots
    if np.any(over):
        for t in truncate_order:
            if not np.any(over):
                break
            excess = np.where(over, running_total - n_slots, 0)
            reduction = np.minimum(excess, keep[:, t])
            keep[:, t] -= reduction
            running_total -= reduction
            over = running_total > n_slots
        if np.any(over):
            raise ValueError(
                f"{int(over.sum())} event(s) still exceed n_slots={n_slots} after "
                f"truncating every type to zero -- n_slots is too small for this "
                f"analysis's real multiplicities, not a truncation-policy problem."
            )

    # --- Step 2: lay out contiguously in pack_order ---
    x = np.zeros((N, n_slots, n_features), dtype=np.float32)
    x_mask = np.zeros((N, n_slots), dtype=bool)
    slot_start = np.zeros(N, dtype=np.int64)

    for t in pack_order:
        k_t = keep[:, t]                 # (N,) how many of this type survive, per event
        feats = type_features[t]         # (N, k_max_t, F), already within-type sorted
        k_max_t = feats.shape[1]

        for j in range(k_max_t):
            active = j < k_t              # does this event have a surviving j-th particle of this type
            dest = slot_start + j
            valid = active & (dest < n_slots)
            if not np.any(valid):
                continue
            rows = np.nonzero(valid)[0]
            x[rows, dest[rows], :] = feats[rows, j, :]
            x_mask[rows, dest[rows]] = True

        slot_start = slot_start + k_t

    return x, x_mask


if __name__ == "__main__":
    # Self-test -- run directly (`python3 pack_slots.py`) before trusting this
    # as a template. Three events: normal case, an empty type, and a jet
    # count that forces truncation.
    n_slots = 6  # small on purpose, to make truncation easy to hand-check

    # jets: up to 4 per event, feature = [1000 + jet_index] so placement is checkable
    jet_feats = np.zeros((3, 4, 1), dtype=np.float32)
    jet_feats[0, :2, 0] = [1000, 1001]   # event 0: 2 real jets
    jet_feats[1, :0, 0] = []             # event 1: 0 real jets
    jet_feats[2, :4, 0] = [1000, 1001, 1002, 1003]  # event 2: 4 real jets (will need truncation)
    jet_counts = np.array([2, 0, 4])

    # leptons: up to 3 per event, feature = [2000 + lepton_index]
    lep_feats = np.zeros((3, 3, 1), dtype=np.float32)
    lep_feats[0, :2, 0] = [2000, 2001]   # event 0: 2 leptons
    lep_feats[1, :1, 0] = [2000]         # event 1: 1 lepton
    lep_feats[2, :3, 0] = [2000, 2001, 2002]  # event 2: 3 leptons

    lep_counts = np.array([2, 1, 3])

    x, x_mask = pack_slots(
        type_features=[jet_feats, lep_feats],
        type_counts=[jet_counts, lep_counts],
        pack_order=[0, 1],        # jets first, then leptons
        truncate_order=[0, 1],    # truncate jets first
        n_slots=n_slots,
        n_features=1,
    )

    # Event 0: 2 jets + 2 leptons = 4, fits in 6 -> [1000,1001,2000,2001,0,0]
    expect0 = [1000, 1001, 2000, 2001, 0, 0]
    got0 = x[0, :, 0].tolist()
    assert got0 == expect0, f"event 0 mismatch: {got0} != {expect0}"
    assert x_mask[0].tolist() == [True, True, True, True, False, False]

    # Event 1: 0 jets + 1 lepton -> lepton lands at slot 0, not slot 4
    expect1 = [2000, 0, 0, 0, 0, 0]
    got1 = x[1, :, 0].tolist()
    assert got1 == expect1, f"event 1 mismatch: {got1} != {expect1}"
    assert x_mask[1].tolist() == [True, False, False, False, False, False]

    # Event 2: 4 jets + 3 leptons = 7 > 6 -> truncate 1 jet (lowest priority
    # jet, i.e. the pT-sorted tail -- here that's whichever the caller put
    # last in jet_feats' own within-type order, index 3, value 1003), keep
    # all 3 leptons: [1000,1001,1002, 2000,2001,2002]
    expect2 = [1000, 1001, 1002, 2000, 2001, 2002]
    got2 = x[2, :, 0].tolist()
    assert got2 == expect2, f"event 2 mismatch: {got2} != {expect2}"
    assert x_mask[2].tolist() == [True, True, True, True, True, True]

    print("Small hand-checkable self-test passed.")

    # ------------------------------------------------------------------
    # Larger randomized invariant test at realistic scale (18 slots, 7
    # features, 200k events, with a deliberately high overflow/truncation
    # rate) -- the small test above only exercises truncation on a single
    # hand-picked event; this checks the vectorized truncation path holds
    # up under real load before this file gets trusted for an actual
    # (expensive, GPU-bound) data-conversion run.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(1234)
    N = 200_000
    N_SLOTS_BIG = 18
    N_FEAT_BIG = 7

    JET_MAX = 22  # deliberately > n_slots so jets alone can force truncation
    jet_counts = rng.integers(0, JET_MAX + 1, size=N)
    jet_feats = np.zeros((N, JET_MAX, N_FEAT_BIG), dtype=np.float32)
    LEP_MAX = 4
    lep_counts = rng.integers(0, LEP_MAX + 1, size=N)
    lep_feats = np.zeros((N, LEP_MAX, N_FEAT_BIG), dtype=np.float32)
    for i in range(N):
        nj = jet_counts[i]
        if nj:
            jet_feats[i, :nj, 1] = np.sort(rng.uniform(20, 500, size=nj))[::-1]  # feature[1]=pt, descending
            jet_feats[i, :nj, 5] = 0.0  # feature[5]=isLepton
        nl = lep_counts[i]
        if nl:
            lep_feats[i, :nl, 1] = np.sort(rng.uniform(10, 200, size=nl))[::-1]
            lep_feats[i, :nl, 5] = 1.0

    x_big, mask_big = pack_slots(
        type_features=[jet_feats, lep_feats],
        type_counts=[jet_counts, lep_counts],
        pack_order=[0, 1],
        truncate_order=[0, 1],
        n_slots=N_SLOTS_BIG,
        n_features=N_FEAT_BIG,
    )

    combined_raw = jet_counts + lep_counts
    expected_real = np.minimum(combined_raw, N_SLOTS_BIG)
    assert np.array_equal(mask_big.sum(axis=1), expected_real), \
        "large-scale test: real-slot count per event does not match min(jets+leptons, n_slots)"

    n_overflow = int((combined_raw > N_SLOTS_BIG).sum())
    mismatches = 0
    for i in rng.choice(N, size=2000, replace=False):
        nj, nl = jet_counts[i], lep_counts[i]
        if nj + nl > N_SLOTS_BIG:
            nj_eff = min(nj, max(0, N_SLOTS_BIG - nl))  # jets truncated first
            nl_eff = nl
        else:
            nj_eff, nl_eff = nj, nl
        row, m = x_big[i], mask_big[i]
        if not (np.all(row[:nj_eff, 5] == 0.0) and np.all(m[:nj_eff])):
            mismatches += 1
        if not (np.all(row[nj_eff:nj_eff + nl_eff, 5] == 1.0) and np.all(m[nj_eff:nj_eff + nl_eff])):
            mismatches += 1
        if np.any(m[nj_eff + nl_eff:]):
            mismatches += 1
    assert mismatches == 0, f"large-scale test: {mismatches} placement mismatches found"

    print(f"Large-scale randomized self-test passed: N={N}, {n_overflow} events "
          f"({100 * n_overflow / N:.1f}%) required truncation, 0 mismatches on a 2000-event spot-check.")
