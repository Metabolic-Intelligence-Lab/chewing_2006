# Swallow detection & chewing–swallowing coordination — analysis pipeline

Focused, reproducible pipeline built on a **validated 1:1 port of the lab's official
algorithm** (`analisi_chewing.py` → `src/official.py`; reproduces
`database_finale_completo.xlsx` with r=1.000, 100% exact match). The scope is
deliberately narrow:

1. a **swallow-detection pipeline** — deglutition events read off the submental
   (mylohyoid) channel despite its co-activation during chewing, via phase-end
   windows + a morphology filter — with its validation, and
2. the **chewing↔swallowing coordination** this second channel makes measurable.

## Run
```
pip install -r requirements.txt
python analysis/run_all.py            # full pipeline -> outputs/ + report/ docx
python analysis/verify_manuscript.py  # check every manuscript number vs the tables
```
The consolidated manuscript docx is built automatically by `run_all.py` (final step,
via `make_article.build()`). Cohort: adults ≥18 (n=84); swallowing/coordination use
valid-swallow subjects (n=64). Seeds fixed (reproducible).

## Layout
```
analisi_chewing.py   original lab script (reference algorithm)
analysis/src/
  official.py      faithful port of analisi_chewing.py (phases, deglutition via
                   phase-end windows + morphology filter, work/work-rate/cycle)
  io_raw.py        raw txt parsing, real-time axis, fs estimate
  features.py      heuristic burst/swallow detector (used only for QC & validation)
  coordination.py  coordination_official(): chew<->swallow metrics from official events
  merge.py         master_dataset.csv = official features + coordination + metadata
  cohort.py        adults() / adults_valid_swallow() — single cohort definition
  stats.py         coordination statistics (Focus A only): Spearman + bootstrap CI, FDR
  swallowing.py    swallowing descriptive deep-dive (descriptors + distributions)
  swallow_morphology.py per-event swallow shape (rise/decay/RFD/sub-peaks) + %ref norm
  event_level.py   within-meal single-swallow morphology dynamics (mixed-effects)
  cycles.py        single-chew extraction (~5347), within-meal dynamics, rhythm/spectrum
  directionality.py  chew<->swallow spectral coherence + bidirectional Granger causality
  validate_swallows.py  borderline contact sheets (official vs heuristic) + agreement [PRIMARY validation]
  sensitivity.py   one-at-a-time sweep of detection params: count/ranking robustness [PRIMARY validation]
  annotation_concordance.py  concordance of detected events vs the manual protocol
                   annotations (phases/*.txt), wall-clock aligned [SUPPORT validation]
  internal_validation.py  internal/construct validation without a gold standard:
                   in-scope (phase-end window) concordance, chance-level nulls,
                   semi-synthetic sensitivity in the operating regime, plausibility of
                   accepted vs rejected candidates, task-isolated pilot [CONSTRUCT validation]
  confound_checks.py  zero-swallow selection-bias check
  segmentation.py  shared cycle/stroke segmentation helpers
  statutil.py      shared statistics helpers (FDR, effect sizes, bootstrap CIs)
  qc.py            QC report + official-port validation vs the official file
  figures.py       publication figures (example trace, pipeline, morphology,
                   coupling, annotation concordance)
  docx_helpers.py  python-docx helpers + demographics table
  make_article.py  assembles the consolidated MAIN + SUPPLEMENTARY manuscript (single docx)
analysis/run_all.py         end-to-end orchestrator
analysis/verify_manuscript.py  manuscript <-> tables numeric consistency gate
analysis/outputs/
  master_dataset.csv, official_features.csv, swallow_events.csv, chew_cycles.csv
  qc_report.md, qc_port_validation.csv
  tables/ (A_coordination_*, SW_*, VAL_*, SENS_*, ANN_concordance_*, IV_*, DIR_*,
           CYC_within_sequence, EV_within_meal, CONF_selection_bias)
  figures/ (fig_*, qc_*, validation/borderline_*)
analysis/report/
  Chewing_Swallowing_pipeline_EN.docx      <- consolidated manuscript (main + supplementary)
```

## Data
`dati_raw/` + `phases/` (main cohort, 113 recordings) and `validazione-algoritmo/`
(task-isolated pilot: 2 subjects, `S-K_chew.txt` / `S-K_swal.txt` chew-only / swallow-only
sessions, headerless, no phase log) are subject data and are gitignored. In the pilot,
sessions 2-2 and 2-4 are excluded automatically (overlapping timestamps between the chew
and swal file, flat chew file); the flags are computed in `internal_validation.py`, not
hard-coded.

## Key facts
- **Detection targets TERMINAL (phase-end) swallows.** The gate only looks from 1 s
  before to 8 s after a chewing-phase end, so only ~29% of manual marks are in scope
  (~20% after reaction-time correction); of those the detector recovers ~64% (~68%
  corrected), vs ~28% of all marks. Semi-synthetic swallows injected inside the window
  at chewing-level amplitude are recovered ~74% of the time (≈0% mid-phase, by design).
  Official vs amplitude-heuristic count agreement r≈0.66; per-subject count ranking
  stable to detection params (Spearman ≥0.87) except the chewing-pause threshold
  (ρ≈0.55) and adaptive-threshold percentile (ρ≈0.72), which mainly rescale counts.
- **Event–mark concordance is chance-limited.** ~67% of gated events coincide with a
  manual mark at ±10 s, but chance (circular-shift null) is ~60% — enrichment only
  ×1.1, rising to ×1.4 at ±3 s. Accepted events sit on a quieter masseter than the
  preceding phase in ~81% of cases (median ratio >1 for rejected in-window candidates).
- **Zero-swallow ≠ no swallowing.** 0 gated swallows in 28/113 recordings (~24%;
  20/84 adults), yet all 28 carried manual annotations (168 swallows, only ~19 of them
  in a phase-end window) — swallows during ongoing chewing, gated out, not absent.
  Absolute swallow counts are terminal-swallow counts = lower bounds.
- **Annotation concordance is SUPPORT only.** Manual marks are operator button-presses
  (can be missed/delayed; offset median ≈−1.4 s = reaction time); used for concordance,
  not as ground truth. Aligned in absolute wall-clock time (phases log and raw recording
  have different zero references).
- **Task-isolated pilot (n=2) is a construct check.** Without the phase-end gate the
  candidate+morphology stages fire ~22 events/min on chew-only vs ~5/min on
  swallow-only sessions: morphology alone does not reject chewing; the gate does.
- **Chewing↔swallowing coupling.** Power positively coupled (ρ=0.37, p_FDR=0.014);
  envelopes synchronous (cross-correlation lag ≈0 s); directionality symmetric
  (coherence median ≈0.49; Granger both ways ~93–94%, net≈0) — mylohyoid co-activated
  during chewing, not sequentially driven.
- **Within-meal.** Cycle-level (5347 chews): chews shorten/sharpen through the meal
  (rise time β=−0.05/SD, p=0.0002; duration β=−0.03/SD, p=0.03), chewing freq ~1.2–1.6 Hz.
  Single-swallow morphology stable across the meal (event-limited).
- **Single-swallow morphology** (276 events): median duration ≈1.9 s, rise ≈0.85 s,
  FWHM ≈0.85 s, 2 sub-peaks (piecemeal) — genuine swallow bursts, not chewing spikes.
- Amplitude is uncalibrated (arbitrary units) — all intensity readings cohort-relative.
```

## License
Code and documentation are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license — see `LICENSE`.
