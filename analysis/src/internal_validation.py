"""Internal / construct validation of the (v1) terminal-swallow detector.

No instrumental swallow ground truth exists for this dataset. The only
reference is the operator's button-press (delayed, missable). This module
therefore does NOT claim accuracy; it asks five narrower, answerable
questions, each of which can be checked without a gold standard:

  A. IN-SCOPE CONCORDANCE. The detector is gated to a window around the end
     of each chewing phase (-1 s .. +8 s), i.e. it targets TERMINAL swallows
     by construction. What fraction of the manual marks fall inside that
     operating window, and what is the recall restricted to them (with and
     without the median reaction-time correction)?
  B. CHANCE-LEVEL NULL. Marks are dense (~3/min) and the match tolerance is
     wide (+-10 s), so a random event lands near a mark often. Two nulls:
     circular shift of the marks (``shift``) and re-placement of the detected
     events uniformly inside their own allowed windows (``in_window``,
     gate-aware, conservative). Enrichment = observed / null mean.
  C. SEMI-SYNTHETIC SENSITIVITY IN THE OPERATING REGIME. Inject a template
     swallow at KNOWN times inside phase-end windows, at a range of amplitudes
     relative to the local chewing contamination, and measure the recall of
     the unmodified official detector.
  D. PHYSIOLOGICAL PLAUSIBILITY. Enumerate every candidate the detector saw
     (accepted and rejected, with the first failing gate) and compare
     masseter ratio, FWHM and duration between groups.
  E. TASK-ISOLATED PILOT (n=2 subjects). Chew-only vs swallow-only sessions
     recorded without a phase log; run the candidate + morphology stages
     ungated and report events/min per task. Provenance flags are computed,
     not hard-coded.

Outputs (tables/): IV_inscope_subject.csv, IV_inscope_summary.csv,
IV_null_concordance.csv, IV_semisynthetic_recall.csv, IV_semisynthetic_fp.csv,
IV_candidates.csv, IV_plausibility_summary.csv, IV_pilot_files.csv,
IV_pilot_summary.csv.  Figures: fig_internal_validation.png,
fig_pilot_task_isolated.png.
"""
from __future__ import annotations

import os
import pathlib
import re

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu

import official
import io_raw
import annotation_concordance as ac
import swallow_morphology as sm

FS = float(official.SAMPLING_RATE)
REACTION_OFFSET_S = 1.4          # default; overridden from ANN_concordance_summary.csv
TOLS = (2.0, 3.0, 5.0, 10.0)
N_NULL = 1000
SEED = 20260829
ALPHAS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
LIT_BURST_S = (0.5, 2.0)         # literature submental burst duration range
SEMI_MATCH_TOL_S = 1.0
PILOT_FILE_RX = re.compile(r"(\d+)-(\d+)_(chew|swal)\.txt$")
PILOT_FLAT_P95 = 60.0            # a.u.: chew file with masseter p95 below this is flat
PILOT_ACTIVE_START = 50.0        # a.u.: mean of first 0.5 s above this = starts mid-activity
PILOT_MIN_DELAY_S = 1.0          # pilot files are 18-76 s: official 5 s delay is too costly


# ---------------------------------------------------------------- templates
def swallow_template(fs=FS, rise_s=0.85, decay_s=1.05):
    """Raised-cosine swallow burst (rise 0.85 s, decay 1.05 s), peak = 1."""
    nr, nd = max(2, int(rise_s * fs)), max(2, int(decay_s * fs))
    rise = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nr)))
    decay = 0.5 * (1 + np.cos(np.linspace(0, np.pi, nd)))
    g = np.concatenate([rise, decay[1:]])
    return g / g.max()


def _event_peak_times(events, deg, fs=FS):
    """Peak time (s) of each detected (start, end) event."""
    ts = []
    for s, e in events:
        seg = deg[s:e]
        pk = s + (int(np.argmax(seg)) if len(seg) else 0)
        ts.append(pk / fs)
    return np.array(ts)


# ---------------------------------------------------------------- matching
def _match_float(a_s, e_s, tol_s):
    """Greedy one-to-one match on float seconds (same rule as ac._match)."""
    a_s, e_s = np.asarray(a_s, float), np.asarray(e_s, float)
    if len(a_s) == 0 or len(e_s) == 0:
        return 0, []
    off = e_s[None, :] - a_s[:, None]
    ai, ei = np.where(np.abs(off) <= tol_s)
    order = np.argsort(np.abs(off[ai, ei]), kind="stable")
    used_a, used_e, offsets = set(), set(), []
    for k in order:
        a, e = ai[k], ei[k]
        if a in used_a or e in used_e:
            continue
        used_a.add(a); used_e.add(e); offsets.append(float(off[a, e]))
    return len(offsets), offsets


def _in_zones_s(t, zones_s):
    return any(s <= t <= e for s, e in zones_s)


def _reaction_offset(out_dir) -> float:
    p = pathlib.Path(out_dir) / "tables" / "ANN_concordance_summary.csv"
    if p.exists():
        v = pd.read_csv(p).get("offset_median_s")
        if v is not None and len(v) and v.iloc[0] == v.iloc[0]:
            # offset = detected - annotation (negative: mark trails the event)
            return float(-v.iloc[0])
    return REACTION_OFFSET_S


# ---------------------------------------------------------------- context
def _load_context(raw_dir, phases_dir, official_res=None):
    """One dict per recording: signals, phases, events, allowed windows (s),
    annotations in seconds from the first raw sample (wall-clock aligned)."""
    res = official_res or official.analyze_all(raw_dir)
    raw_paths = {io_raw._parse_id(p): p for p in io_raw.list_recordings(raw_dir)}
    ann_idx = ac._phases_index(phases_dir)
    ctx = []
    for rid, (_summary, ex) in res.items():
        if rid not in raw_paths:
            continue
        rec = io_raw.load_recording(raw_paths[rid])
        if rec.n_samples == 0 or len(rec.timestamps) == 0:
            continue
        first_ts = rec.timestamps.iloc[0]
        annots = []
        if rid in ann_idx:
            _t0, ann = ac.parse_annotations(ann_idx[rid])
            annots = [(a - first_ts).total_seconds() for a in ann]
        deg = np.asarray(ex["deg"], float)
        allowed = official.build_allowed_zones(ex["phases"], len(deg))
        ctx.append(dict(
            rid=int(rid), mast=np.asarray(ex["mast"], float), deg=deg,
            phases=list(ex["phases"]), events=list(ex["events"]),
            mast_threshold=float(ex["mast_threshold"]), deg_start=int(ex["deg_start"]),
            allowed=allowed, allowed_s=[(s / FS, e / FS) for s, e in allowed],
            ev_s=np.array([s / FS for s, _e in ex["events"]], float),
            annots=np.array(annots, float), has_ann=rid in ann_idx,
            T_s=len(deg) / FS, first_ts=first_ts,
        ))
    ctx.sort(key=lambda c: c["rid"])
    # self-check: float matcher == Timestamp matcher on the first annotated recording
    for c in ctx:
        if len(c["annots"]) and len(c["ev_s"]):
            base = pd.Timestamp("2000-01-01")
            a_ts = [base + pd.Timedelta(seconds=float(a)) for a in c["annots"]]
            e_ts = [base + pd.Timedelta(seconds=float(e)) for e in c["ev_s"]]
            n_ref, _ = ac._match(a_ts, e_ts, ac.MATCH_TOL_S)
            n_new, _ = _match_float(c["annots"], c["ev_s"], ac.MATCH_TOL_S)
            assert n_ref == n_new, f"matcher mismatch on {c['rid']}: {n_ref} vs {n_new}"
            break
    return ctx


# ---------------------------------------------------------------- A
def in_scope_concordance(ctx, reaction_offset_s=REACTION_OFFSET_S, tols=(5.0, 10.0)):
    rows = []
    for c in ctx:
        if not c["has_ann"]:
            continue
        a = c["annots"]
        insc = np.array([_in_zones_s(t, c["allowed_s"]) for t in a], bool)
        insc_c = np.array([_in_zones_s(t - reaction_offset_s, c["allowed_s"]) for t in a], bool)
        row = dict(ID=c["rid"], n_annotated=len(a), n_in_scope=int(insc.sum()),
                   n_in_scope_corr=int(insc_c.sum()), n_detected=len(c["ev_s"]),
                   n_annot_before_deg_start=int((a < c["deg_start"] / FS).sum()))
        for tol in tols:
            k = f"tol{int(tol)}"
            row[f"n_matched_all_{k}"] = _match_float(a, c["ev_s"], tol)[0]
            row[f"n_matched_in_scope_{k}"] = _match_float(a[insc], c["ev_s"], tol)[0]
            row[f"n_matched_in_scope_corr_{k}"] = _match_float(a[insc_c], c["ev_s"], tol)[0]
        rows.append(row)
    sub = pd.DataFrame(rows)
    tot_ann = int(sub.n_annotated.sum())
    summ = dict(n_recordings=len(sub), total_annotated=tot_ann,
                total_in_scope=int(sub.n_in_scope.sum()),
                total_in_scope_corr=int(sub.n_in_scope_corr.sum()),
                frac_annot_in_scope=sub.n_in_scope.sum() / tot_ann,
                frac_annot_in_scope_corr=sub.n_in_scope_corr.sum() / tot_ann,
                reaction_offset_s=reaction_offset_s)
    for tol in tols:
        k = f"tol{int(tol)}"
        summ[f"recall_all_{k}"] = sub[f"n_matched_all_{k}"].sum() / tot_ann
        summ[f"recall_in_scope_{k}"] = sub[f"n_matched_in_scope_{k}"].sum() / max(1, sub.n_in_scope.sum())
        summ[f"recall_in_scope_corr_{k}"] = (sub[f"n_matched_in_scope_corr_{k}"].sum()
                                             / max(1, sub.n_in_scope_corr.sum()))
    zero = sub[sub.n_detected == 0]
    summ.update(zero_detection_recordings=len(zero),
                zero_detection_annotations=int(zero.n_annotated.sum()),
                zero_detection_in_scope_annotations=int(zero.n_in_scope.sum()),
                zero_detection_in_scope_corr_annotations=int(zero.n_in_scope_corr.sum()))
    return sub, pd.DataFrame([summ])


# ---------------------------------------------------------------- B
def null_concordance(ctx, n_null=N_NULL, tols=TOLS, seed=SEED,
                     reaction_offset_s=REACTION_OFFSET_S):
    rng = np.random.default_rng(seed)
    cs = [c for c in ctx if c["has_ann"]]   # same recording set as ANN_concordance
    tot_ann = sum(len(c["annots"]) for c in cs)
    tot_det = sum(len(c["ev_s"]) for c in cs)
    tot_insc = sum(int(np.sum([_in_zones_s(t, c["allowed_s"]) for t in c["annots"]])) for c in cs)

    def pooled(a_list, e_list, tol):
        return sum(_match_float(a, e, tol)[0] for a, e in zip(a_list, e_list))

    obs = {tol: pooled([c["annots"] for c in cs], [c["ev_s"] for c in cs], tol) for tol in tols}
    obs_insc = {tol: pooled([c["annots"][[_in_zones_s(t, c["allowed_s"]) for t in c["annots"]]]
                             for c in cs], [c["ev_s"] for c in cs], tol) for tol in tols}
    null_shift = {tol: [] for tol in tols}
    null_shift_insc = {tol: [] for tol in tols}
    null_shift_insc_n = []
    null_win = {tol: [] for tol in tols}
    for _ in range(n_null):
        a_sh, a_sh_insc, e_win = [], [], []
        n_insc = 0
        for c in cs:
            sh = (c["annots"] + rng.uniform(0, c["T_s"])) % c["T_s"]
            a_sh.append(sh)
            m = np.array([_in_zones_s(t, c["allowed_s"]) for t in sh], bool)
            a_sh_insc.append(sh[m]); n_insc += int(m.sum())
            if len(c["ev_s"]) and c["allowed_s"]:
                lens = np.array([e - s for s, e in c["allowed_s"]])
                w = rng.choice(len(lens), size=len(c["ev_s"]), p=lens / lens.sum())
                e_win.append(np.array([rng.uniform(*c["allowed_s"][i]) for i in w]))
            else:
                e_win.append(c["ev_s"])
        null_shift_insc_n.append(n_insc)
        for tol in tols:
            null_shift[tol].append(pooled(a_sh, [c["ev_s"] for c in cs], tol))
            null_shift_insc[tol].append(pooled(a_sh_insc, [c["ev_s"] for c in cs], tol))
            null_win[tol].append(pooled([c["annots"] for c in cs], e_win, tol))

    rows = []
    def add(null, metric, tol, obs_val, null_vals, denom_obs, denom_null):
        nv = np.asarray(null_vals, float) / np.asarray(denom_null, float)
        ov = obs_val / denom_obs
        rows.append(dict(null=null, metric=metric, tol_s=tol, observed_count=obs_val,
                         observed=ov, null_mean=float(nv.mean()),
                         null_ci_lo=float(np.percentile(nv, 2.5)),
                         null_ci_hi=float(np.percentile(nv, 97.5)),
                         enrichment=ov / nv.mean() if nv.mean() > 0 else np.nan,
                         p_value=(int(np.sum(nv >= ov)) + 1) / (len(nv) + 1),
                         n_null=len(nv)))
    for tol in tols:
        add("shift", "specificity", tol, obs[tol], null_shift[tol], tot_det, tot_det)
        add("shift", "recall", tol, obs[tol], null_shift[tol], tot_ann, tot_ann)
        add("shift", "recall_in_scope", tol, obs_insc[tol], null_shift_insc[tol],
            max(1, tot_insc), np.maximum(1, null_shift_insc_n))
        add("in_window", "specificity", tol, obs[tol], null_win[tol], tot_det, tot_det)
        add("in_window", "recall", tol, obs[tol], null_win[tol], tot_ann, tot_ann)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C
def _phase_end_slots(phases, n, fs, dur_n, deg_start=0, pre_s=1.0, post_s=8.0,
                     next_phase_guard_s=3.0):
    """Injection onsets inside the operating window of each phase end, with
    room for the full template and clear of the next phase's start buffer."""
    slots = []
    pre, post = int(pre_s * fs), int(post_s * fs)
    for i, (_ps, pe) in enumerate(phases):
        lo = max(deg_start, pe - pre)
        hi = min(n - dur_n - 1, pe + post - dur_n)
        if i + 1 < len(phases):
            hi = min(hi, phases[i + 1][0] - dur_n - int(next_phase_guard_s * fs))
        if hi > lo:
            slots.append((lo, hi))
    return slots


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def semisynthetic_operating_regime(ctx, alphas=ALPHAS, per_subject=3, n_subjects=None,
                                   seed=SEED, match_tol_s=SEMI_MATCH_TOL_S, guard_s=3.0):
    g = swallow_template()
    dur_n = len(g)
    rng = np.random.default_rng(seed)
    cand = [c for c in ctx if len(c["phases"]) >= 1 and len(c["mast"]) > 6 * FS]
    if n_subjects:
        cand = cand[:n_subjects]
    rec = {a: [0, 0] for a in alphas}      # hits, injected
    used = 0
    win_events = win_min = out_events = out_min = 0.0
    for c in cand:
        mast, deg, phases = c["mast"], c["deg"], c["phases"]
        slots = _phase_end_slots(phases, len(deg), FS, dur_n, c["deg_start"])
        if not slots:
            continue
        ev_pk = _event_peak_times(c["events"], deg) * FS
        # existing (un-injected) detections in/out of window: FP proxy + sanity
        inwin = np.zeros(len(deg), bool)
        for s, e in c["allowed"]:
            inwin[s:e + 1] = True
        win_min += inwin.sum() / FS / 60.0
        out_min += (~inwin).sum() / FS / 60.0
        for pk in ev_pk:
            (win_events, out_events)  # noqa
            if inwin[min(int(pk), len(deg) - 1)]:
                win_events += 1
            else:
                out_events += 1
        inphase = np.zeros(len(deg), bool)
        for s, e in phases:
            inphase[s:e] = True
        pos = deg[inphase][deg[inphase] > 0]
        base_level = max(float(np.median(pos)) if len(pos) else 1.0, 1.0)
        onsets = []
        for _ in range(per_subject):
            for _try in range(50):
                lo, hi = slots[rng.integers(len(slots))]
                o = int(rng.integers(lo, hi))
                pk = o + int(0.85 * FS)
                if len(ev_pk) == 0 or np.min(np.abs(ev_pk - pk)) > guard_s * FS:
                    onsets.append(o); break
        if not onsets:
            continue
        used += 1
        for a in alphas:
            deg_inj = deg.copy()
            pk_t = []
            for o in onsets:
                deg_inj[o:o + dur_n] += g * a * base_level
                pk_t.append((o + int(0.85 * FS)) / FS)
            ev, *_ = official.identify_deglutition_events(deg_inj, mast, phases)
            t_det = _event_peak_times(ev, deg_inj)
            for pt in pk_t:
                rec[a][1] += 1
                if len(t_det) and np.min(np.abs(t_det - pt)) <= match_tol_s:
                    rec[a][0] += 1
    rows = []
    for a in alphas:
        k, n = rec[a]
        lo, hi = _wilson(k, n)
        rows.append(dict(alpha=a, n_injected=n, n_recordings=used, n_recovered=k,
                         recall_v1=k / n if n else np.nan, recall_ci_lo=lo, recall_ci_hi=hi))
    fp = pd.DataFrame([
        dict(region="phase_end_window", events=int(win_events), minutes=round(win_min, 2),
             events_per_min=win_events / win_min if win_min else np.nan),
        dict(region="outside_window", events=int(out_events), minutes=round(out_min, 2),
             events_per_min=out_events / out_min if out_min else np.nan),
    ])
    return pd.DataFrame(rows), fp


# ---------------------------------------------------------------- D
def enumerate_candidates(deg, mast, phases, fs=FS, env_filter_size=151, env_peak_height=8,
                         env_prominence=5, env_peak_distance=80, border_fraction=0.10,
                         mast_filter_size=51, adaptive_percentile=25, min_sustained_s=1.0,
                         min_delay_after_start_s=5.0, phase_start_buffer_s=3.0,
                         use_phase_end_windows=True, pre_phase_end_s=1.0, post_phase_end_s=8.0,
                         mast_relax_factor=1.50, max_derivative_sign_changes_per_s=35.0,
                         max_derivative_activity_ratio=4.00, min_deg_amplitude=10,
                         min_duration_s=0.15, max_duration_s=5.0, fine_filter_size=7):
    """Replica of official.identify_deglutition_events that keeps EVERY
    candidate with its gate outcomes instead of only the accepted events.
    Fidelity is asserted by the caller (accepted set == official events)."""
    deg = np.asarray(deg, float); mast = np.asarray(mast, float)
    analysis_start, deg_start, mast_threshold = official.compute_start_and_threshold(
        mast, fs, mast_filter_size, adaptive_percentile, min_sustained_s,
        min_delay_after_start_s=min_delay_after_start_s)
    deg_env = median_filter(deg, size=env_filter_size)
    sm_mast = median_filter(mast, size=mast_filter_size)
    deg_fine = median_filter(deg, size=fine_filter_size)
    excluded = official.build_excluded_zones(deg_start, phases or [], fs, phase_start_buffer_s)
    allowed = (official.build_allowed_zones(phases or [], len(deg), fs, pre_phase_end_s,
                                            post_phase_end_s) if use_phase_end_windows else [])
    peaks, _ = find_peaks(deg_env, height=env_peak_height, distance=env_peak_distance,
                          prominence=env_prominence)
    peaks = peaks[peaks >= deg_start]
    cols = ["peak_idx", "left", "right", "dur_s", "max_amp", "sc_per_s", "dar", "local_mast",
            "mast_threshold", "in_allowed", "in_excluded", "deriv_ok", "mast_ok", "amp_ok",
            "accepted", "reject_reason"]
    if len(peaks) == 0:
        return pd.DataFrame(columns=cols), sm_mast
    valleys = [int(np.argmin(deg_env[peaks[i]:peaks[i + 1]])) + peaks[i]
               for i in range(len(peaks) - 1)]
    rows = []
    for j, pk in enumerate(peaks):
        lb = (max(deg_start, pk - int(2.0 * fs)) if j == 0 else valleys[j - 1])
        rb = (min(len(deg) - 1, pk + int(2.0 * fs)) if j == len(peaks) - 1 else valleys[j])
        half = deg_env[pk] * border_fraction
        left = pk
        while left > lb and deg_env[left] > half:
            left -= 1
        right = pk
        while right < rb and deg_env[right] > half:
            right += 1
        if right - left < int(min_duration_s * fs):
            left = max(deg_start, pk - int(0.3 * fs))
            right = min(len(deg) - 1, pk + int(0.5 * fs))
        dur = (right - left) / fs
        max_amp = float(deg[left:right].max()) if right > left else 0.0
        in_exc = bool(official._in(pk, excluded) or official._overlaps(left, right, excluded))
        in_all = bool(official._in_incl(pk, allowed) or official._overlaps_allowed(left, right, allowed))
        seg = deg_fine[left:right]
        d_seg = np.diff(seg)
        if len(d_seg):
            sc = np.sign(d_seg); sc = sc[sc != 0]
            sc_per_s = float(np.sum(sc[1:] != sc[:-1]) / dur) if len(sc) > 1 else 0.0
            mean_amp = float(np.mean(seg))
            dar = float(np.mean(np.abs(d_seg)) / mean_amp) if mean_amp > 0 else 0.0
        else:
            sc_per_s, dar = np.nan, np.nan
        deriv_ok = bool(len(d_seg) and sc_per_s <= max_derivative_sign_changes_per_s
                        and dar <= max_derivative_activity_ratio)
        local_mast = float(sm_mast[left:right].mean()) if right > left else np.nan
        mast_ok = bool(local_mast < mast_threshold * mast_relax_factor
                       or official._in_incl(pk, allowed))
        amp_ok = max_amp >= min_deg_amplitude
        dur_ok = min_duration_s <= dur <= max_duration_s
        reason = None
        if in_exc:
            reason = "excluded_zone"
        elif use_phase_end_windows and not in_all:
            reason = "outside_window"
        elif not dur_ok:
            reason = "duration"
        elif len(d_seg) == 0:
            reason = "empty"
        elif not amp_ok:
            reason = "amplitude"
        elif not deriv_ok:
            reason = "morphology"
        elif not mast_ok:
            reason = "masseter"
        rows.append(dict(peak_idx=int(pk), left=int(left), right=int(right), dur_s=dur,
                         max_amp=max_amp, sc_per_s=sc_per_s, dar=dar, local_mast=local_mast,
                         mast_threshold=mast_threshold, in_allowed=in_all, in_excluded=in_exc,
                         deriv_ok=deriv_ok, mast_ok=mast_ok, amp_ok=amp_ok,
                         accepted=reason is None, reject_reason=reason or "accepted"))
    return pd.DataFrame(rows, columns=cols), sm_mast


def plausibility(ctx):
    """Per-candidate table + grouped summary (accepted vs rejected in/out window)."""
    frames, mismatches = [], 0
    for c in ctx:
        cand, sm_mast = enumerate_candidates(c["deg"], c["mast"], c["phases"])
        acc = set((int(l), int(r)) for l, r in cand.loc[cand.accepted, ["left", "right"]].itertuples(index=False))
        off = set((int(l), int(r)) for l, r in c["events"])
        if acc != off:
            mismatches += 1
        if cand.empty:
            continue
        deg, mast = c["deg"], c["mast"]
        ref = sm.channel_reference(deg)
        ctx_n = int(3 * FS)
        extra = []
        for r in cand.itertuples(index=False):
            l, rr = int(r.left), int(r.right)
            ef = sm.event_features(deg, l, rr, FS, ref) if rr - l >= 3 else {}
            # phase reference: last phase that ends at/before the peak, else containing it
            ph = [p for p in c["phases"] if p[1] <= r.peak_idx] or \
                 [p for p in c["phases"] if p[0] <= r.peak_idx <= p[1]]
            ph_mast = float(sm_mast[ph[-1][0]:ph[-1][1]].mean()) if ph else np.nan
            ctx_seg = np.concatenate([sm_mast[max(0, l - ctx_n):l], sm_mast[rr:rr + ctx_n]])
            ctx_mast = float(ctx_seg.mean()) if len(ctx_seg) else np.nan
            m_pk = float(mast[l:rr].max()) if rr > l else np.nan
            extra.append(dict(
                fwhm_s=ef.get("fwhm_s", np.nan), rise_time_s=ef.get("rise_time_s", np.nan),
                mast_ratio_phase=r.local_mast / ph_mast if ph_mast and ph_mast > 0 else np.nan,
                mast_ratio_context=r.local_mast / ctx_mast if ctx_mast and ctx_mast > 0 else np.nan,
                mast_below_thr=bool(r.local_mast < r.mast_threshold),
                sub_to_mast_peak_ratio=r.max_amp / max(m_pk, 1.0) if m_pk == m_pk else np.nan))
        df = pd.concat([cand.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
        df.insert(0, "ID", c["rid"])
        frames.append(df)
    cands = pd.concat(frames, ignore_index=True)
    cands["group"] = np.where(cands.accepted, "accepted",
                              np.where(cands.reject_reason.isin(["excluded_zone", "outside_window"]),
                                       "rejected_out_of_window", "rejected_in_window"))
    metrics = ["dur_s", "fwhm_s", "rise_time_s", "max_amp", "mast_ratio_phase",
               "mast_ratio_context", "sub_to_mast_peak_ratio", "sc_per_s", "dar"]
    rows = []
    for m in metrics:
        acc = cands.loc[cands.group == "accepted", m].dropna()
        rin = cands.loc[cands.group == "rejected_in_window", m].dropna()
        p_in = mannwhitneyu(acc, rin).pvalue if len(acc) > 2 and len(rin) > 2 else np.nan
        for gname, g in cands.groupby("group"):
            v = g[m].dropna()
            rows.append(dict(metric=m, group=gname, n=len(v),
                             median=float(v.median()) if len(v) else np.nan,
                             q1=float(v.quantile(.25)) if len(v) else np.nan,
                             q3=float(v.quantile(.75)) if len(v) else np.nan,
                             p_mwu_accepted_vs_rejected_in_window=p_in))
    acc = cands[cands.group == "accepted"]
    lo, hi = LIT_BURST_S
    frac = dict(metric="fractions", group="accepted", n=len(acc),
                frac_fwhm_in_lit_range=float(((acc.fwhm_s >= lo) & (acc.fwhm_s <= hi)).mean()),
                frac_dur_in_lit_range=float(((acc.dur_s >= lo) & (acc.dur_s <= hi)).mean()),
                frac_mast_ratio_phase_lt1=float((acc.mast_ratio_phase < 1).mean()),
                frac_mast_below_thr=float(acc.mast_below_thr.mean()),
                frac_sub_gt_mast_peak=float((acc.sub_to_mast_peak_ratio > 1).mean()),
                self_check_mismatched_recordings=mismatches,
                n_accepted_total=int(cands.accepted.sum()))
    summ = pd.concat([pd.DataFrame(rows), pd.DataFrame([frac])], ignore_index=True)
    reasons = cands.reject_reason.value_counts().rename_axis("reject_reason").reset_index(name="n")
    return cands, summ, reasons, mismatches


# ---------------------------------------------------------------- E
def adjust_signal_robust(sig, q=5.0):
    """Pilot-only baseline: subtract the q-th percentile (robust to files that
    start mid-activity, where official.adjust_signal's first-0.5 s mean would
    zero the signal). Clipped at 0 like the official variant."""
    sig = np.asarray(sig, float)
    out = sig - np.percentile(sig, q)
    out[out < 0] = 0
    return out


def _pilot_timestamps(path):
    first = last = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            ts = pd.to_datetime(f"{parts[0]} {parts[1]}", errors="coerce")
            if pd.isna(ts):
                continue
            if first is None:
                first = ts
            last = ts
    return first, last


def load_pilot_file(path):
    m = PILOT_FILE_RX.search(os.path.basename(path))
    if not m:
        return None
    arr = official.load_and_preprocess_data(path)
    if arr.size == 0:
        return None
    first, last = _pilot_timestamps(path)
    return dict(subject=int(m.group(1)), session=int(m.group(2)), task=m.group(3),
                pair=f"{int(m.group(1))}-{int(m.group(2))}", mast_raw=arr[:, 0],
                deg_raw=arr[:, 1], first_ts=first, last_ts=last, n=len(arr), dur_s=len(arr) / FS)


def _pilot_detect(mast, deg, mode):
    if mode == "official":
        _a, _d, thr = official.compute_start_and_threshold(mast)
        phases = official.identify_mastication_phases(mast, thr)
        ev, *_ = official.identify_deglutition_events(deg, mast, phases)
        return ev, len(phases)
    ev, *_ = official.identify_deglutition_events(
        deg, mast, [], use_phase_end_windows=False, min_delay_after_start_s=PILOT_MIN_DELAY_S)
    return ev, 0


def pilot_task_isolated(pilot_dir):
    files = [load_pilot_file(os.path.join(pilot_dir, f)) for f in sorted(os.listdir(pilot_dir))]
    files = [f for f in files if f]
    rows, traces = [], {}
    for f in files:
        mast0, deg0 = official.adjust_signal(f["mast_raw"]), official.adjust_signal(f["deg_raw"])
        mast5, deg5 = adjust_signal_robust(f["mast_raw"]), adjust_signal_robust(f["deg_raw"])
        n05 = int(0.5 * FS)
        ev_off, n_ph = _pilot_detect(mast0, deg0, "official")
        ev_un, _ = _pilot_detect(mast0, deg0, "ungated")
        ev_un5, _ = _pilot_detect(mast5, deg5, "ungated")
        mp95, sp95 = float(np.percentile(mast5, 95)), float(np.percentile(deg5, 95))
        rows.append(dict(
            file=f"{f['pair']}_{f['task']}", subject=f["subject"], session=f["session"],
            pair=f["pair"], task=f["task"], dur_s=round(f["dur_s"], 1),
            first_ts=f["first_ts"], last_ts=f["last_ts"],
            mast_p95=mp95, sub_p95=sp95, sub_to_mast_p95_ratio=sp95 / max(mp95, 1.0),
            start_mean_0_5s=float(max(f["mast_raw"][:n05].mean(), f["deg_raw"][:n05].mean())),
            n_phases=n_ph, n_events_official=len(ev_off), n_events_ungated=len(ev_un),
            n_events_ungated_p5=len(ev_un5),
            events_per_min_ungated=len(ev_un) / f["dur_s"] * 60,
            events_per_min_ungated_p5=len(ev_un5) / f["dur_s"] * 60))
        traces[f"{f['pair']}_{f['task']}"] = (mast5, deg5, ev_un5)
    df = pd.DataFrame(rows)
    # provenance flags (computed)
    flags, reasons = [], []
    for r in df.itertuples(index=False):
        fl = []
        mate = df[(df.pair == r.pair) & (df.task != r.task)]
        if len(mate):
            m = mate.iloc[0]
            if r.first_ts is not None and m.first_ts is not None and \
               r.first_ts <= m.last_ts and m.first_ts <= r.last_ts:
                fl.append("overlap")
        if r.task == "chew" and r.mast_p95 < PILOT_FLAT_P95:
            fl.append("flat_chew")
        if r.start_mean_0_5s > PILOT_ACTIVE_START:
            fl.append("starts_active")
        flags.append(";".join(fl)); reasons.append(bool(fl))
    df["flag"] = flags
    df["flagged"] = reasons
    # pair-level inclusion: excluded if overlap or flat chew anywhere in the pair
    bad = set(df.loc[df.flag.str.contains("overlap|flat_chew"), "pair"])
    df["included"] = ~df.pair.isin(bad)
    prs = []
    for pair, g in df.groupby("pair"):
        ch, sw = g[g.task == "chew"], g[g.task == "swal"]
        if not len(ch) or not len(sw):
            continue
        ch, sw = ch.iloc[0], sw.iloc[0]
        prs.append(dict(pair=pair, included=bool(ch.included),
                        flags=";".join(sorted(set(filter(None, [ch.flag, sw.flag])))),
                        chew_dur_s=ch.dur_s, swal_dur_s=sw.dur_s,
                        chew_events_per_min=ch.events_per_min_ungated_p5,
                        swal_events_per_min=sw.events_per_min_ungated_p5,
                        chew_events_per_min_first05=ch.events_per_min_ungated,
                        swal_events_per_min_first05=sw.events_per_min_ungated,
                        chew_sub_to_mast=ch.sub_to_mast_p95_ratio,
                        swal_sub_to_mast=sw.sub_to_mast_p95_ratio,
                        chew_official_events=ch.n_events_official,
                        swal_official_events=sw.n_events_official))
    pairs = pd.DataFrame(prs)
    inc = pairs[pairs.included]
    pooled = dict(pair="POOLED_INCLUDED", included=True, flags="",
                  chew_dur_s=inc.chew_dur_s.sum(), swal_dur_s=inc.swal_dur_s.sum(),
                  chew_events_per_min=(df[df.included & (df.task == "chew")].n_events_ungated_p5.sum()
                                       / max(inc.chew_dur_s.sum(), 1) * 60),
                  swal_events_per_min=(df[df.included & (df.task == "swal")].n_events_ungated_p5.sum()
                                       / max(inc.swal_dur_s.sum(), 1) * 60),
                  chew_events_per_min_first05=(df[df.included & (df.task == "chew")].n_events_ungated.sum()
                                               / max(inc.chew_dur_s.sum(), 1) * 60),
                  swal_events_per_min_first05=(df[df.included & (df.task == "swal")].n_events_ungated.sum()
                                               / max(inc.swal_dur_s.sum(), 1) * 60),
                  chew_sub_to_mast=inc.chew_sub_to_mast.median(),
                  swal_sub_to_mast=inc.swal_sub_to_mast.median(),
                  chew_official_events=inc.chew_official_events.sum(),
                  swal_official_events=inc.swal_official_events.sum(),
                  n_pairs_included=len(inc), n_pairs_total=len(pairs),
                  n_subjects=df.subject.nunique())
    pairs = pd.concat([pairs, pd.DataFrame([pooled])], ignore_index=True)
    return df, pairs, traces


# ---------------------------------------------------------------- figures
def make_figures(out_dir, traces=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = pathlib.Path(out_dir)
    T, F = out / "tables", out / "figures"
    F.mkdir(parents=True, exist_ok=True)
    insc = pd.read_csv(T / "IV_inscope_summary.csv").iloc[0]
    null = pd.read_csv(T / "IV_null_concordance.csv")
    semi = pd.read_csv(T / "IV_semisynthetic_recall.csv")
    cands = pd.read_csv(T / "IV_candidates.csv")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
    # (a) recall: all vs in-scope vs in-scope corrected, with shift-null band at +-10 s
    a = ax[0, 0]
    vals = [insc.recall_all_tol10, insc.recall_in_scope_tol10, insc.recall_in_scope_corr_tol10]
    a.bar(["all marks", "in-scope", "in-scope\n(reaction-corr.)"], vals,
          color=["#9E9E9E", "#1F6FB2", "#5B8C5A"])
    nr = null[(null.null == "shift") & (null.metric == "recall") & (null.tol_s == 10)].iloc[0]
    a.axhspan(nr.null_ci_lo, nr.null_ci_hi, color="#C0504D", alpha=.15, label="shift-null 95% (all marks)")
    a.set_ylim(0, 1); a.set_ylabel("fraction of manual marks recovered (±10 s)")
    a.set_title("(a) Recall inside the operating window"); a.legend(fontsize=8, loc="upper left")
    for i, v in enumerate(vals):
        a.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9)
    # (b) enrichment vs tolerance
    b = ax[0, 1]
    for nm, col in (("shift", "#1F6FB2"), ("in_window", "#E8A33D")):
        s = null[(null.null == nm) & (null.metric == "specificity")].sort_values("tol_s")
        b.plot(s.tol_s, s.enrichment, "-o", color=col, label=f"{nm} null")
    b.axhline(1, color="k", lw=.8, ls="--")
    b.set_xlabel("match tolerance (s)"); b.set_ylabel("observed / chance (events matched)")
    b.set_title("(b) Concordance enrichment over chance"); b.legend(fontsize=8)
    # (c) semi-synthetic recall vs alpha
    c = ax[1, 0]
    c.errorbar(semi.alpha, semi.recall_v1, yerr=[semi.recall_v1 - semi.recall_ci_lo,
                                                  semi.recall_ci_hi - semi.recall_v1],
               fmt="-o", color="#1F6FB2", capsize=3, label="phase-end window (operating regime)")
    mid = T / "V2_semisynthetic_recovery.csv"
    if mid.exists():
        m = pd.read_csv(mid)
        c.plot(m.alpha, m.recall_v1, "--s", color="#9E9E9E", label="mid-phase (out of scope)")
    c.set_ylim(-.03, 1.03); c.set_xlabel("injected peak / local chewing level on submental channel")
    c.set_ylabel("recall of injected swallows (±1 s)")
    c.set_title("(c) Semi-synthetic sensitivity, v1 unchanged"); c.legend(fontsize=8)
    # (d) plausibility: masseter ratio and FWHM by group
    d = ax[1, 1]
    order = ["accepted", "rejected_in_window", "rejected_out_of_window"]
    data = [cands.loc[cands.group == g, "mast_ratio_phase"].dropna().clip(upper=3) for g in order]
    bp = d.boxplot(data, positions=[1, 2, 3], widths=.5, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], ["#5B8C5A", "#E8A33D", "#9E9E9E"]):
        patch.set_facecolor(col); patch.set_alpha(.6)
    d.axhline(1, color="k", lw=.8, ls="--")
    d.set_xticks([1, 2, 3]); d.set_xticklabels(["accepted", "rejected\nin window", "rejected\nout of window"])
    d.set_ylabel("masseter during candidate / preceding phase")
    d.set_title("(d) Masseter context of candidates")
    fig.tight_layout(); fig.savefig(F / "fig_internal_validation.png", dpi=150); plt.close(fig)

    if traces:
        pf = pd.read_csv(T / "IV_pilot_files.csv").set_index("file")
        keys = list(traces)
        fig, axes = plt.subplots(len(keys), 1, figsize=(11, 1.7 * len(keys)), sharex=False)
        for axx, k in zip(np.atleast_1d(axes), keys):
            mast, deg, ev = traces[k]
            t = np.arange(len(deg)) / FS
            inc = bool(pf.loc[k, "included"])
            axx.plot(t, mast, color="#9E9E9E" if not inc else "#C0504D", lw=.6, label="masseter")
            axx.plot(t, deg, color="#9E9E9E" if not inc else "#1F6FB2", lw=.6, label="submental")
            for s, e in ev:
                axx.axvspan(s / FS, e / FS, color="#5B8C5A", alpha=.25)
            tag = f"{k}  ({'included' if inc else 'excluded: ' + str(pf.loc[k, 'flag'])})"
            axx.set_title(tag, fontsize=9, loc="left"); axx.set_ylabel("a.u.", fontsize=8)
        np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("time (s)")
        fig.tight_layout(); fig.savefig(F / "fig_pilot_task_isolated.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- run
def run(raw_dir, phases_dir, pilot_dir, out_dir, official_res=None, n_null=N_NULL) -> dict:
    T = pathlib.Path(out_dir) / "tables"
    T.mkdir(parents=True, exist_ok=True)
    off = _reaction_offset(out_dir)
    ctx = _load_context(raw_dir, phases_dir, official_res)

    sub, summ = in_scope_concordance(ctx, off)
    sub.to_csv(T / "IV_inscope_subject.csv", index=False)
    summ.to_csv(T / "IV_inscope_summary.csv", index=False)
    s = summ.iloc[0]
    print(f"      in-scope: {s.frac_annot_in_scope:.1%} of marks in window "
          f"({s.frac_annot_in_scope_corr:.1%} corrected by {off:.1f} s); recall in-scope "
          f"{s.recall_in_scope_tol10:.2f} (corr {s.recall_in_scope_corr_tol10:.2f}) vs all "
          f"{s.recall_all_tol10:.2f} @±10 s; zero-detection recordings: "
          f"{s.zero_detection_in_scope_annotations}/{s.zero_detection_annotations} marks in scope")

    null = null_concordance(ctx, n_null=n_null, reaction_offset_s=off)
    null.to_csv(T / "IV_null_concordance.csv", index=False)
    r10 = null[(null.null == "shift") & (null.metric == "specificity") & (null.tol_s == 10)].iloc[0]
    r3 = null[(null.null == "shift") & (null.metric == "specificity") & (null.tol_s == 3)].iloc[0]
    print(f"      null: spec obs {r10.observed:.3f} vs chance {r10.null_mean:.3f} "
          f"(x{r10.enrichment:.2f}, p={r10.p_value:.3f}) @±10 s; x{r3.enrichment:.2f} @±3 s")

    semi, fp = semisynthetic_operating_regime(ctx)
    semi.to_csv(T / "IV_semisynthetic_recall.csv", index=False)
    fp.to_csv(T / "IV_semisynthetic_fp.csv", index=False)
    print("      semi-synthetic recall (operating regime): " +
          ", ".join(f"α={r.alpha:g}: {r.recall_v1:.2f}" for r in semi.itertuples()))

    cands, plaus, reasons, mism = plausibility(ctx)
    cands.to_csv(T / "IV_candidates.csv", index=False)
    plaus.to_csv(T / "IV_plausibility_summary.csv", index=False)
    reasons.to_csv(T / "IV_reject_reasons.csv", index=False)
    fr = plaus[plaus.metric == "fractions"].iloc[0]
    print(f"      candidates: {len(cands)} ({int(fr.n_accepted_total)} accepted; self-check "
          f"mismatched recordings = {mism}); FWHM in {LIT_BURST_S}: {fr.frac_fwhm_in_lit_range:.0%}; "
          f"masseter ratio<1: {fr.frac_mast_ratio_phase_lt1:.0%}")
    if mism:
        raise RuntimeError(f"enumerate_candidates drifted from official on {mism} recordings")

    traces = None
    headline = dict(reaction_offset_s=off, inscope=s.to_dict(), null_spec10=r10.to_dict(),
                    null_spec3=r3.to_dict(), semi=semi, plaus=fr.to_dict())
    if pilot_dir and os.path.isdir(pilot_dir):
        pf, pp, traces = pilot_task_isolated(pilot_dir)
        pf.to_csv(T / "IV_pilot_files.csv", index=False)
        pp.to_csv(T / "IV_pilot_summary.csv", index=False)
        pooled = pp[pp.pair == "POOLED_INCLUDED"].iloc[0]
        print(f"      pilot: {int(pooled.n_pairs_included)}/{int(pooled.n_pairs_total)} pairs "
              f"included ({int(pooled.n_subjects)} subjects); ungated events/min chew-only "
              f"{pooled.chew_events_per_min:.1f} vs swallow-only {pooled.swal_events_per_min:.1f}; "
              f"flagged: {', '.join(pf[pf.flagged].file + '[' + pf[pf.flagged].flag + ']')}")
        headline["pilot"] = pooled.to_dict()
    else:
        print("      pilot: directory not found - task-isolated pilot skipped")
    make_figures(out_dir, traces)
    return headline


if __name__ == "__main__":
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    run(str(base / "dati_raw"), str(base / "phases"), str(base / "validazione-algoritmo"), str(out))
