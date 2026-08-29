"""End-to-end pipeline (official-algorithm basis).

Focused scope: the swallowing DETECTION pipeline and chewing<->swallowing
COORDINATION. QC + port validation -> master dataset (official features +
coordination) -> coordination statistics -> swallowing descriptive deep-dive ->
swallow morphology + within-meal dynamics -> detection validation (heuristic
agreement + parameter sensitivity + protocol-annotation concordance) ->
directionality -> confound checks -> figures -> manuscript.

Run:  python analysis/run_all.py
Outputs under analysis/outputs/. Reproducible (seeds fixed).
"""
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
SRC = BASE / "src"
OUT = BASE / "outputs"
sys.path.insert(0, str(SRC))

RAW = str(ROOT / "dati_raw")
PHASES = str(ROOT / "phases")
PILOT = str(ROOT / "validazione-algoritmo")   # task-isolated pilot (private, untracked)
DB = str(ROOT / "database_finale_completo.xlsx")

import qc
import merge
import stats
import swallowing
import validate_swallows
import swallow_morphology
import event_level
import sensitivity
import annotation_concordance
import internal_validation
import directionality
import cycles
import confound_checks
import figures


_STEP = 0


def step(msg):
    """Print a sequentially-numbered pipeline step."""
    global _STEP
    _STEP += 1
    print(f"[{_STEP}] {msg}")


def main():
    OUT.mkdir(exist_ok=True)
    step("QC + official-port validation ...")
    qc.run(RAW, DB, str(OUT))
    step("Master dataset (official features + coordination) ...")
    df = merge.build_master(RAW, DB)
    df.to_csv(OUT / "master_dataset.csv", index=False)
    print(f"      master {df.shape}; adults={int(df.is_adult.sum())}, "
          f"valid_swallow={int(df.valid_swallow.sum())}, "
          f"adults+valid={int((df.is_adult & df.valid_swallow).sum())}")
    mc = str(OUT / "master_dataset.csv")
    step("Coordination statistics (chew<->swallow) ...")
    stats.run(mc, str(OUT))
    step("Swallowing descriptive deep-dive ...")
    swallowing.run(mc, str(OUT))
    step("Swallow morphology (event-level features) ...")
    ev = swallow_morphology.build_event_table(RAW)
    ev.to_csv(OUT / "swallow_events.csv", index=False)
    print(f"      {len(ev)} swallow events, {ev.ID.nunique()} subjects")
    step("Event-level mixed models + within-meal dynamics ...")
    event_level.run(RAW, mc, str(OUT))
    step("Cycle-level extraction + within-sequence + rhythm ...")
    cyc, rhythm, ws = cycles.run(RAW, mc, str(OUT))
    print(f"       {len(cyc)} chewing cycles extracted")
    step("Detection validation - heuristic-agreement contact sheets ...")
    validate_swallows.run(RAW, mc, str(OUT))
    step("Detection validation - parameter sensitivity ...")
    sensitivity.run(RAW, mc, str(OUT))
    step("Detection validation - protocol-annotation concordance (support) ...")
    annotation_concordance.run(RAW, PHASES, str(OUT))
    step("Detection validation - internal/construct validation (in-scope, null, "
         "semi-synthetic, plausibility, task-isolated pilot) ...")
    internal_validation.run(RAW, PHASES, PILOT, str(OUT))
    step("Chew<->swallow directionality (coherence + Granger) ...")
    directionality.run(RAW, str(OUT))
    step("Confound checks (zero-swallow selection bias) ...")
    confound_checks.run(mc, str(OUT / "chew_cycles.csv"), str(OUT))
    step("Figures ...")
    figures.run(RAW, mc, str(OUT / "figures"))
    step("Consolidated manuscript (single docx: article + supplementary) ...")
    import make_article
    make_article.build()
    print("Done ->", OUT)


if __name__ == "__main__":
    main()
