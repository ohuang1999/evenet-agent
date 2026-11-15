#!/usr/bin/env python3
"""
ROOT ntuple -> EveNet .npz converter for the soft-photon study (lab-frame, 2nd
iteration: more particles, charge feature, mass replaces energy in slot 0).

Channel
-------
    Bs1(5830) -> B*0 KS0,  B*0 -> gamma + B0,  B0 -> J/psi K*0(892) -> mu mu K pi,
    KS0 -> pi pi.

What's new vs lab_1stTry/make_npz_softgamma_split.py
----------------------------------------------------
* `x` now packs 10 physical particles (still in the 18-slot pretrain layout):
    slot 0: mu1   (mu+)
    slot 1: mu2   (mu-)
    slot 2: K     (from K*0 sub-pair: trk_rf[0..1] with m > 400 MeV)
    slot 3: Pi    (the other one in trk_rf[0..1])
    slot 4: Pi1   (pi+ from KS0 sub-pair: trk_rf[2..3] with trk_charge > 0)
    slot 5: Pi2   (pi- from KS0 sub-pair)
    slot 6: Jpsi  (from jpsi_rf_*)
    slot 7: Kst   (K + Pi, composite)
    slot 8: Ks    (ld_rf_* with PDG mass 497.611)
    slot 9: B0    (jx_rf_* with PDG mass 5279.66)
    slot 10..17 : padding (mask=0, features=0)
* Per-slot 7-feature layout (pretrain-shape preserved):
    [0]=energy       (replaces "energy" in pretrain layout)
    [1]=pT
    [2]=eta
    [3]=phi
    [4]=0          (was btag; unused)
    [5]=isLepton   (1 for muons, 0 otherwise)
    [6]=charge     (mu_charge / trk_charge, 0 for composites)
* Target: lab-frame truth photon (pT, eta, phi) -> truth_gam_*
* Event selection: only events with match_all == 1 are kept.

Output .npz contents
--------------------
    x                      (N, 18, 7)  float32
    x_mask                 (N, 18)     bool
    num_vectors            (N,)        float32   10 + 1 (global) = 11
    num_sequential_vectors (N,)        float32   10
    conditions             (N, 10)     float32   zeros
    conditions_mask        (N, 1)      bool      all True
    classification         (N,)        int64     all 0 ("signal")
    event_weight           (N,)        float32   all 1
    x_invisible            (N, 1, 3)   float32   [pT, eta, phi] of truth gamma
    x_invisible_mask       (N, 1)      bool      photon validity
    num_invisible_raw      (N,)        int64
    num_invisible_valid    (N,)        int64

Usage
-----
    pip install uproot awkward numpy
    python make_npz_softgamma_split.py \
        --input  /path/to/signal*.root \
        --tree   YourTreeName \
        --output /path/to/lab_2stTry/signal.npz \
        --compress
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SLOTS_X      = 18                       # pretrain layout: keep 18 slots
N_FEATURES_X   = 7                        # pretrain layout: keep 7 features
N_INVISIBLE    = 1
N_FEATURES_INV = 3                        # pretrain Neutrinos schema (pt, eta, phi)
N_CONDITIONS   = 10                       # pretrain GLOBAL.Conditions width

N_MU  = 2                                 # 2 reco muons
N_TRK = 4                                 # 4 reco tracks (K + pi(K*) + 2x pi(KS0))
N_PARTICLES = 10                          # mu1,mu2,K,Pi,Pi1,Pi2,Jpsi,Kst,Ks,B0

# Slot indices for the 10 used positions
S_MU1, S_MU2, S_K, S_PI, S_PI1, S_PI2, S_JPSI, S_KST, S_KS, S_B0 = range(N_PARTICLES)

# PDG masses (MeV) for the composite B0 and KS0
PDG_B0_M = 5279.66
PDG_KS_M = 497.611

# K vs pi separation in the K*0 sub-pair (trk_rf_m[0..1])
K_MASS_THR = 400.0                        # MeV (any value between m_pi and m_K)

assert N_PARTICLES <= N_SLOTS_X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_2d(arr: ak.Array, n: int) -> np.ndarray:
    """Convert a uproot 'array of length n' branch to a (N, n) numpy.float32.
    Works for both regular and jagged storage; clips/pads to exactly n."""
    counts = ak.num(arr, axis=1).to_numpy()
    padded = ak.pad_none(arr, n, clip=True)
    return ak.fill_none(padded, 0.0).to_numpy().astype(np.float32), counts


def _to_1d(arr: ak.Array) -> np.ndarray:
    return ak.to_numpy(arr).astype(np.float32)


def _ptetaphim_to_pxpypze(pt, eta, phi, m):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p  = pt * np.cosh(eta)
    E  = np.sqrt(p * p + m * m)
    return px, py, pz, E


def _pxpypze_to_ptetaphim(px, py, pz, E):
    pt   = np.sqrt(px * px + py * py)
    p2   = pt * pt + pz * pz
    safe = pt > 0.0
    eta  = np.where(safe, np.arcsinh(pz / np.where(safe, pt, 1.0)), 0.0)
    phi  = np.arctan2(py, px)
    m2   = E * E - p2
    m    = np.sqrt(np.clip(m2, 0.0, None))
    return pt, eta, phi, m


def _gather(arr2d: np.ndarray, idx_1d: np.ndarray) -> np.ndarray:
    """Per-event 1D gather: arr2d (N, K), idx_1d (N,) -> (N,)."""
    return np.take_along_axis(arr2d, idx_1d[:, None].astype(np.int64),
                              axis=1).squeeze(1)


# ---------------------------------------------------------------------------
# Branches to read
# ---------------------------------------------------------------------------
RECO_BRANCHES = [
    # muons (length 2)
    "mu_rf_pt", "mu_rf_eta", "mu_rf_phi", "mu_rf_m",
    "mu_charge",
    # tracks (length 4)
    "trk_rf_pt", "trk_rf_eta", "trk_rf_phi", "trk_rf_m",
    "trk_charge",
    # composites (scalars)
    "jpsi_rf_pt", "jpsi_rf_eta", "jpsi_rf_phi", "jpsi_rf_m",
    "jx_rf_pt",   "jx_rf_eta",   "jx_rf_phi",            # B0 (mass forced to PDG)
    "ld_rf_pt",   "ld_rf_eta",   "ld_rf_phi",            # KS0 (mass forced to PDG)
]
TRUTH_BRANCHES = [
    "truth_gam_pt", "truth_gam_eta", "truth_gam_phi",
]
SELECTION_BRANCHES = ["match_all"]
BRANCHES = RECO_BRANCHES + TRUTH_BRANCHES + SELECTION_BRANCHES


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def convert_tree(tree, max_events: int | None = None) -> dict:
    arr = tree.arrays(BRANCHES, library="ak", entry_stop=max_events)
    N0 = len(arr)

    # ---------------------------------------------------------------------
    # 1) Selection: keep only match_all == 1
    # ---------------------------------------------------------------------
    match_all = ak.to_numpy(arr["match_all"]).astype(np.int64)
    keep = match_all == 1
    n_kept    = int(keep.sum())
    n_dropped = int(N0 - n_kept)
    print(f"    match_all selection: kept {n_kept} / {N0} "
          f"(dropped {n_dropped}, {100.0 * n_dropped / max(1, N0):.2f} %)")

    arr = arr[keep]
    N = len(arr)
    if N == 0:
        raise SystemExit("No events left after match_all==1 cut.")

    # ---------------------------------------------------------------------
    # 2) Read reco arrays
    # ---------------------------------------------------------------------
    mu_pt , _ = _to_2d(arr["mu_rf_pt"],  N_MU)
    mu_eta, _ = _to_2d(arr["mu_rf_eta"], N_MU)
    mu_phi, _ = _to_2d(arr["mu_rf_phi"], N_MU)
    mu_m  , _ = _to_2d(arr["mu_rf_m"],   N_MU)
    mu_q  , _ = _to_2d(arr["mu_charge"], N_MU)

    trk_pt , _ = _to_2d(arr["trk_rf_pt"],  N_TRK)
    trk_eta, _ = _to_2d(arr["trk_rf_eta"], N_TRK)
    trk_phi, _ = _to_2d(arr["trk_rf_phi"], N_TRK)
    trk_m  , _ = _to_2d(arr["trk_rf_m"],   N_TRK)
    trk_q  , _ = _to_2d(arr["trk_charge"], N_TRK)

    jpsi_pt  = _to_1d(arr["jpsi_rf_pt"])
    jpsi_eta = _to_1d(arr["jpsi_rf_eta"])
    jpsi_phi = _to_1d(arr["jpsi_rf_phi"])
    jpsi_m   = _to_1d(arr["jpsi_rf_m"])

    jx_pt  = _to_1d(arr["jx_rf_pt"])
    jx_eta = _to_1d(arr["jx_rf_eta"])
    jx_phi = _to_1d(arr["jx_rf_phi"])
    # B0 mass forced to PDG
    jx_m   = np.full(N, PDG_B0_M, dtype=np.float32)

    ld_pt  = _to_1d(arr["ld_rf_pt"])
    ld_eta = _to_1d(arr["ld_rf_eta"])
    ld_phi = _to_1d(arr["ld_rf_phi"])
    # KS0 mass forced to PDG
    ld_m   = np.full(N, PDG_KS_M, dtype=np.float32)

    # ---------------------------------------------------------------------
    # 3) Choose mu1 (mu+) and mu2 (mu-)
    # ---------------------------------------------------------------------
    # mu_pos_idx ∈ {0, 1}: which branch index is the positive muon.
    mu_pos_idx = np.where(mu_q[:, 0] > 0, 0, 1).astype(np.int64)
    mu_neg_idx = 1 - mu_pos_idx

    mu1_pt  = _gather(mu_pt , mu_pos_idx); mu1_eta = _gather(mu_eta, mu_pos_idx)
    mu1_phi = _gather(mu_phi, mu_pos_idx); mu1_m   = _gather(mu_m  , mu_pos_idx)
    mu1_q   = _gather(mu_q  , mu_pos_idx)

    mu2_pt  = _gather(mu_pt , mu_neg_idx); mu2_eta = _gather(mu_eta, mu_neg_idx)
    mu2_phi = _gather(mu_phi, mu_neg_idx); mu2_m   = _gather(mu_m  , mu_neg_idx)
    mu2_q   = _gather(mu_q  , mu_neg_idx)

    # ---------------------------------------------------------------------
    # 4) Choose K and Pi from the K*0 sub-pair (trk_rf[0..1] by mass > 400)
    # ---------------------------------------------------------------------
    k_idx  = np.where(trk_m[:, 0] > K_MASS_THR, 0, 1).astype(np.int64)
    pi_idx = 1 - k_idx

    K_pt  = _gather(trk_pt , k_idx); K_eta = _gather(trk_eta, k_idx)
    K_phi = _gather(trk_phi, k_idx); K_m   = _gather(trk_m  , k_idx)
    K_q   = _gather(trk_q  , k_idx)

    Pi_pt  = _gather(trk_pt , pi_idx); Pi_eta = _gather(trk_eta, pi_idx)
    Pi_phi = _gather(trk_phi, pi_idx); Pi_m   = _gather(trk_m  , pi_idx)
    Pi_q   = _gather(trk_q  , pi_idx)

    # ---------------------------------------------------------------------
    # 5) Choose Pi1 (pi+) and Pi2 (pi-) from the KS0 sub-pair (trk_rf[2..3])
    # ---------------------------------------------------------------------
    # local index within trk_rf[2:4], then add 2 to land on absolute index 2 or 3
    pi1_local = np.where(trk_q[:, 2] > 0, 0, 1).astype(np.int64)
    pi2_local = 1 - pi1_local
    pi1_idx   = 2 + pi1_local
    pi2_idx   = 2 + pi2_local

    Pi1_pt  = _gather(trk_pt , pi1_idx); Pi1_eta = _gather(trk_eta, pi1_idx)
    Pi1_phi = _gather(trk_phi, pi1_idx); Pi1_m   = _gather(trk_m  , pi1_idx)
    Pi1_q   = _gather(trk_q  , pi1_idx)

    Pi2_pt  = _gather(trk_pt , pi2_idx); Pi2_eta = _gather(trk_eta, pi2_idx)
    Pi2_phi = _gather(trk_phi, pi2_idx); Pi2_m   = _gather(trk_m  , pi2_idx)
    Pi2_q   = _gather(trk_q  , pi2_idx)

    # ---------------------------------------------------------------------
    # 6) Build Kst = K + Pi as a composite 4-vector
    # ---------------------------------------------------------------------
    K_px , K_py , K_pz , K_E  = _ptetaphim_to_pxpypze(K_pt , K_eta , K_phi , K_m)
    Pi_px, Pi_py, Pi_pz, Pi_E = _ptetaphim_to_pxpypze(Pi_pt, Pi_eta, Pi_phi, Pi_m)
    Kst_px = K_px + Pi_px
    Kst_py = K_py + Pi_py
    Kst_pz = K_pz + Pi_pz
    Kst_E  = K_E  + Pi_E
    Kst_pt, Kst_eta, Kst_phi, Kst_m = _pxpypze_to_ptetaphim(
        Kst_px, Kst_py, Kst_pz, Kst_E)

    # ---------------------------------------------------------------------
    # 7) Pack into x (N, 18, 7) with the per-slot layout
    #    [mass, pT, eta, phi, 0, isLepton, charge]
    # ---------------------------------------------------------------------
    x      = np.zeros((N, N_SLOTS_X, N_FEATURES_X), dtype=np.float32)
    x_mask = np.zeros((N, N_SLOTS_X),               dtype=bool)

    def _set_slot(slot, m, pt, eta, phi, is_lep, q):
        x[:, slot, 0] = m.astype(np.float32)
        x[:, slot, 1] = pt.astype(np.float32)
        x[:, slot, 2] = eta.astype(np.float32)
        x[:, slot, 3] = phi.astype(np.float32)
        x[:, slot, 4] = 0.0
        x[:, slot, 5] = float(is_lep)
        x[:, slot, 6] = q.astype(np.float32)
        x_mask[:, slot] = True

    _set_slot(S_MU1 , mu1_m, mu1_pt, mu1_eta, mu1_phi, 1, mu1_q)
    _set_slot(S_MU2 , mu2_m, mu2_pt, mu2_eta, mu2_phi, 1, mu2_q)
    _set_slot(S_K   ,   K_m,   K_pt,   K_eta,   K_phi, 0,   K_q)
    _set_slot(S_PI  ,  Pi_m,  Pi_pt,  Pi_eta,  Pi_phi, 0,  Pi_q)
    _set_slot(S_PI1 , Pi1_m, Pi1_pt, Pi1_eta, Pi1_phi, 0, Pi1_q)
    _set_slot(S_PI2 , Pi2_m, Pi2_pt, Pi2_eta, Pi2_phi, 0, Pi2_q)
    _set_slot(S_JPSI, jpsi_m, jpsi_pt, jpsi_eta, jpsi_phi, 0, np.zeros(N, np.float32))
    _set_slot(S_KST , Kst_m, Kst_pt, Kst_eta, Kst_phi, 0, np.zeros(N, np.float32))
    _set_slot(S_KS  ,  ld_m,  ld_pt,  ld_eta,  ld_phi, 0, np.zeros(N, np.float32))
    _set_slot(S_B0  ,  jx_m,  jx_pt,  jx_eta,  jx_phi, 0, np.zeros(N, np.float32))

    # zero out invalid (padding) slots so log_normalize is safe
    x = x * x_mask[..., None].astype(np.float32)

    n_seq                  = x_mask.sum(axis=1).astype(np.float32)
    num_sequential_vectors = n_seq                                 # 10
    num_vectors            = (n_seq + 1.0).astype(np.float32)      # 10 + 1

    # ---------------------------------------------------------------------
    # 8) Global condition tokens (placeholder zeros, pretrain width)
    # ---------------------------------------------------------------------
    conditions      = np.zeros((N, N_CONDITIONS), dtype=np.float32)
    conditions_mask = np.ones( (N, 1),            dtype=bool)
    classification  = np.zeros((N,), dtype=np.int64)
    event_weight    = np.ones( (N,), dtype=np.float32)

    # ---------------------------------------------------------------------
    # 9) Target: truth photon lab-frame (pT, eta, phi)
    # ---------------------------------------------------------------------
    gam_pt  = _to_1d(arr["truth_gam_pt"])
    gam_eta = _to_1d(arr["truth_gam_eta"])
    gam_phi = _to_1d(arr["truth_gam_phi"])

    x_invisible = np.stack([gam_pt, gam_eta, gam_phi], axis=-1)
    x_invisible = x_invisible[:, None, :].astype(np.float32)            # (N, 1, 3)
    inv_valid   = np.isfinite(gam_pt) & (gam_pt > 0)
    x_invisible = x_invisible * inv_valid[:, None, None].astype(np.float32)

    x_invisible_mask    = inv_valid[:, None]
    num_invisible_raw   = inv_valid.astype(np.int64)
    num_invisible_valid = inv_valid.astype(np.int64)

    return dict(
        x=x,
        x_mask=x_mask,
        num_vectors=num_vectors,
        num_sequential_vectors=num_sequential_vectors,
        conditions=conditions,
        conditions_mask=conditions_mask,
        classification=classification,
        event_weight=event_weight,
        x_invisible=x_invisible,
        x_invisible_mask=x_invisible_mask,
        num_invisible_raw=num_invisible_raw,
        num_invisible_valid=num_invisible_valid,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, nargs="+",
                    help="One or more ROOT files and/or shell globs.")
    ap.add_argument("--tree", required=True,
                    help="TTree name inside the ROOT files.")
    ap.add_argument("--output", required=True,
                    help="Output .npz path prefix (will produce *_train.npz "
                         "and *_test.npz).")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Truncate each input file to this many events.")
    ap.add_argument("--compress", action="store_true",
                    help="Use np.savez_compressed (slower write, smaller file).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the train/test shuffle.")
    args = ap.parse_args()

    # Expand each input token: literal file, glob pattern, OR directory
    # (in which case all *.root files under it are picked up).
    files: list[str] = []
    for token in args.input:
        if any(c in token for c in "*?["):
            matched = sorted(glob.glob(token))
            if not matched:
                print(f"[!] Glob matched nothing: {token}")
            files.extend(matched)
        elif Path(token).is_dir():
            matched = sorted(str(p) for p in Path(token).rglob("*.root"))
            if not matched:
                print(f"[!] Directory has no .root files: {token}")
            files.extend(matched)
        else:
            files.append(token)
    files = [f for f in files if Path(f).is_file()]
    if not files:
        raise SystemExit(f"No ROOT files matched any of: {args.input}")
    print(f"[+] {len(files)} input file(s) to read")

    chunks = []
    for f in files:
        print(f"[+] Reading {f} : {args.tree}")
        with uproot.open(f) as fh:
            chunks.append(convert_tree(fh[args.tree], max_events=args.max_events))

    merged = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}

    # ---- shuffle + split (0.8 : 0.2, no val) -------------------------------
    N = merged["x"].shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)

    n_train = int(0.8 * N)
    n_test  = N - n_train

    idx_train = idx[:n_train]
    idx_test  = idx[n_train:]

    def split_dict(data, indices):
        return {k: v[indices] for k, v in data.items()}

    train_data = split_dict(merged, idx_train)
    test_data  = split_dict(merged, idx_test)

    print("\n[+] Split summary (in memory)")
    print(f"      train = {n_train}")
    print(f"      test  = {n_test}")

    # ---- save --------------------------------------------------------------
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if args.compress else np.savez

    train_path = out.with_name(out.stem + "_train.npz")
    test_path  = out.with_name(out.stem + "_test.npz")

    saver(train_path, **train_data)
    saver(test_path,  **test_data)

    def n_events_in_npz(path):
        with np.load(path) as d:
            return d["x"].shape[0]

    n_train_disk = n_events_in_npz(train_path)
    n_test_disk  = n_events_in_npz(test_path)

    print(f"\n[+] Wrote:")
    print(f"      {train_path}   ({n_train_disk} events)")
    print(f"      {test_path}    ({n_test_disk} events)")
    print(f"      total = {n_train_disk + n_test_disk}")

    # ---- final tensor shapes ----------------------------------------------
    print("\n[+] Final tensor shapes (train / test)")
    keys = sorted(train_data.keys())
    name_w = max(len(k) for k in keys)
    for k in keys:
        tr, te = train_data[k], test_data[k]
        print(f"      {k:<{name_w}s} : "
              f"train {str(tr.shape):<14s} {str(tr.dtype):<8s}   "
              f"test {str(te.shape):<14s} {str(te.dtype)}")

    # ---- sanity dump: first 2 events --------------------------------------
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    SLOT_NAME = {
        S_MU1: "mu+",  S_MU2: "mu-",
        S_K:   "K",    S_PI:  "pi(K*)",
        S_PI1: "pi+",  S_PI2: "pi-",
        S_JPSI:"Jpsi", S_KST: "Kst",
        S_KS:  "Ks",   S_B0:  "B0",
    }
    def _dump_event(i: int, label: str):
        print(f"\n  --- event {i}  ({label}) ---")
        print(f"    truth_gam (pT, eta, phi) = {merged['x_invisible'][i, 0]}   "
              f"mask = {bool(merged['x_invisible_mask'][i, 0])}")
        print(f"    {'slot':<5}  {'particle':<8}  {'msk':>3}  "
              f"{'mass':>10}  {'pT':>10}  {'eta':>9}  {'phi':>9}  "
              f"{'isLep':>5}  {'q':>5}")
        for s_idx in range(N_SLOTS_X):
            name = SLOT_NAME.get(s_idx, "(pad)")
            row  = merged["x"][i, s_idx]
            m, pt, eta, phi, _, isLep, q = row
            msk = int(merged["x_mask"][i, s_idx])
            print(f"    {s_idx:<5}  {name:<8}  {msk:>3}  "
                  f"{m:>10.3f}  {pt:>10.3f}  {eta:>9.3f}  {phi:>9.3f}  "
                  f"{isLep:>5.1f}  {q:>5.1f}")

    print("\n[+] First 2 / Last 2 events (original order, pre-shuffle)")
    first_idx = list(range(min(2, N)))
    last_idx  = [N - 2, N - 1] if N >= 2 else []
    # avoid duplicating events if N is small
    for i in first_idx:
        _dump_event(i, "first")
    for i in last_idx:
        if i in first_idx:
            continue
        _dump_event(i, "last")


if __name__ == "__main__":
    main()
