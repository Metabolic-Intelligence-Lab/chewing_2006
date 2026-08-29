"""Supporting concordance of the swallow detector vs the protocol annotations.

During acquisition the operator logged each deglutition as a manual
``PHASE_START,SWALLOW`` button-press in ``phases/NNN_phases.txt``. These marks
are NOT a gold standard: a swallow can be MISSED (no button-press) or the press
can be DELAYED relative to the physiological event. We therefore report only
*concordance*, never ground-truth accuracy:

  * per-subject agreement between the annotated swallow count and the detected
    event count (Spearman + Bland-Altman);
  * the distribution of the signed time offset (detected event - nearest
    annotation) for matched pairs, which itself exposes the operator latency.

Time alignment (important): the phases log and the raw recording use DIFFERENT
zero references (the ``START_RECORDING`` mark can trail the first raw sample by
tens of seconds). Everything is therefore compared in ABSOLUTE wall-clock time:
each detected event index ``s`` is mapped to ``first_raw_timestamp + s/FS`` and
compared against the annotation's ``Orario``. Timestamps have 1 s resolution, so
offsets carry a +-1 s quantisation - fine for the several-second latencies of
interest.

Outputs (tables/):
  ANN_concordance_subject.csv  - one row per recording
  ANN_concordance_summary.csv  - pooled headline numbers
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import official
import io_raw

FS = float(official.SAMPLING_RATE)          # detection uses a fixed 100 Hz
MATCH_TOL_S = 10.0                           # max |offset| to call a pair a match
STRICT_TOL_S = 5.0                           # a tighter reference tolerance


def parse_annotations(path: str):
    """Return (t0, annots) where t0 is the START_RECORDING datetime and annots
    is the sorted list of PHASE_START,SWALLOW datetimes. Parsed line-by-line
    (Dettagli may be empty or contain spaces) mirroring official's tolerant IO.
    """
    t0 = None
    annots = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) < 3 or parts[0].lower() == "data":
                continue
            data, orario, evento = parts[0], parts[1], parts[2]
            dettagli = parts[3] if len(parts) > 3 else ""
            ts = pd.to_datetime(f"{data} {orario}", errors="coerce")
            if pd.isna(ts):
                continue
            if evento == "START_RECORDING" and t0 is None:
                t0 = ts
            elif evento == "PHASE_START" and dettagli.upper().startswith("SWALLOW"):
                annots.append(ts)
    return t0, sorted(annots)


def _match(annots, events, tol_s):
    """One-to-one greedy match (increasing |offset|). Returns (n_matched,
    offsets) where offset = detected_event_time - annotation_time [s]."""
    pairs = []
    for ai, a in enumerate(annots):
        for ei, ev in enumerate(events):
            off = (ev - a).total_seconds()
            if abs(off) <= tol_s:
                pairs.append((abs(off), ai, ei, off))
    pairs.sort()
    used_a, used_e, offsets = set(), set(), []
    for _, ai, ei, off in pairs:
        if ai in used_a or ei in used_e:
            continue
        used_a.add(ai); used_e.add(ei); offsets.append(off)
    return len(offsets), offsets


def _phases_index(phases_dir: str) -> dict[int, str]:
    """Map numeric recording ID -> phases file (handles NNN_phases.txt and the
    one NNN-phases.txt variant)."""
    idx = {}
    for f in sorted(os.listdir(phases_dir)):
        if "phases" not in f or not f.endswith(".txt"):
            continue
        digits = "".join(ch for ch in f.split("phases")[0] if ch.isdigit())
        if digits:
            idx[int(digits)] = os.path.join(phases_dir, f)
    return idx


def build_concordance(raw_dir: str, phases_dir: str, official_res=None):
    """Per-recording concordance table. Detected events come from the official
    detector; annotation times from the phases log; alignment is wall-clock."""
    res = official_res or official.analyze_all(raw_dir)
    raw_paths = {io_raw._parse_id(p): p for p in io_raw.list_recordings(raw_dir)}
    ann_idx = _phases_index(phases_dir)

    rows = []
    for rid, (summary, extra) in res.items():
        if rid not in raw_paths or rid not in ann_idx:
            continue
        rec = io_raw.load_recording(raw_paths[rid])
        if rec.n_samples == 0 or len(rec.timestamps) == 0:
            continue
        first_ts = rec.timestamps.iloc[0]
        # detected event onsets in absolute wall-clock (index / detection FS)
        ev_times = [first_ts + pd.Timedelta(seconds=s / FS)
                    for (s, _e) in extra["events"]]
        _t0, annots = parse_annotations(ann_idx[rid])
        n_ann, n_det = len(annots), len(ev_times)
        n_match, offsets = _match(annots, ev_times, MATCH_TOL_S)
        n_match_strict, _ = _match(annots, ev_times, STRICT_TOL_S)
        rows.append(dict(
            ID=int(rid), n_annotated=n_ann, n_detected=n_det,
            n_matched=n_match, n_matched_strict=n_match_strict,
            frac_annot_matched=(n_match / n_ann) if n_ann else np.nan,
            frac_events_matched=(n_match / n_det) if n_det else np.nan,
            median_offset_s=(float(np.median(offsets)) if offsets else np.nan),
        ))
    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def summarise(sub: pd.DataFrame, out_dir: str) -> dict:
    """Pooled headline numbers + the two CSVs. Returns a dict for the manuscript."""
    tables = pathlib.Path(out_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sub.to_csv(tables / "ANN_concordance_subject.csv", index=False)

    n_rec = len(sub)
    with_ann = sub[sub.n_annotated > 0]
    # count agreement across subjects
    if len(with_ann) >= 3 and with_ann.n_detected.nunique() > 1:
        rho, p = spearmanr(with_ann.n_annotated, with_ann.n_detected)
    else:
        rho, p = np.nan, np.nan
    diff = (sub.n_detected - sub.n_annotated).to_numpy(dtype=float)
    ba_mean = float(np.mean(diff)) if len(diff) else np.nan
    ba_sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    # pooled matched fractions (event/annotation-weighted, not per-subject mean)
    tot_ann = int(sub.n_annotated.sum())
    tot_det = int(sub.n_detected.sum())
    tot_match = int(sub.n_matched.sum())
    tot_match_strict = int(sub.n_matched_strict.sum())
    # pooled offset distribution (recompute pooled median/IQR from per-subject
    # medians as a robust summary)
    offs = sub.median_offset_s.dropna().to_numpy()
    # zero-detection subjects: do annotations still exist there? (speaks to
    # whether zero-swallow is genuine absence vs a detector miss)
    zero_det = sub[sub.n_detected == 0]
    zero_det_with_ann = int((zero_det.n_annotated > 0).sum())
    zero_det_ann_total = int(zero_det.n_annotated.sum())

    summ = dict(
        n_recordings=n_rec,
        n_with_annotations=int(len(with_ann)),
        total_annotated=tot_ann,
        total_detected=tot_det,
        total_matched=tot_match,
        count_spearman_rho=float(rho) if rho == rho else np.nan,
        count_spearman_p=float(p) if p == p else np.nan,
        bland_altman_mean_diff=ba_mean,
        bland_altman_loa_lo=(ba_mean - 1.96 * ba_sd) if ba_sd == ba_sd else np.nan,
        bland_altman_loa_hi=(ba_mean + 1.96 * ba_sd) if ba_sd == ba_sd else np.nan,
        pooled_frac_annot_matched=(tot_match / tot_ann) if tot_ann else np.nan,
        pooled_frac_annot_matched_strict=(tot_match_strict / tot_ann) if tot_ann else np.nan,
        pooled_frac_events_matched=(tot_match / tot_det) if tot_det else np.nan,
        offset_median_s=float(np.median(offs)) if len(offs) else np.nan,
        offset_iqr_lo_s=float(np.percentile(offs, 25)) if len(offs) else np.nan,
        offset_iqr_hi_s=float(np.percentile(offs, 75)) if len(offs) else np.nan,
        zero_detection_subjects=int(len(zero_det)),
        zero_detection_with_annotations=zero_det_with_ann,
        zero_detection_annotations_total=zero_det_ann_total,
        match_tol_s=MATCH_TOL_S,
        strict_tol_s=STRICT_TOL_S,
    )
    pd.DataFrame([summ]).to_csv(tables / "ANN_concordance_summary.csv", index=False)
    return summ


def run(raw_dir: str, phases_dir: str, out_dir: str) -> dict:
    sub = build_concordance(raw_dir, phases_dir)
    summ = summarise(sub, out_dir)
    print(f"      annotation concordance: {summ['n_with_annotations']} recordings "
          f"with marks; count rho={summ['count_spearman_rho']:.2f}; "
          f"pooled matched (annot) {summ['pooled_frac_annot_matched']:.0%} "
          f"@±{MATCH_TOL_S:.0f}s; offset median {summ['offset_median_s']:.1f}s; "
          f"zero-detection subjects with annotations "
          f"{summ['zero_detection_with_annotations']}/{summ['zero_detection_subjects']}")
    return summ


if __name__ == "__main__":
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    run(str(base / "dati_raw"), str(base / "phases"), str(out))
