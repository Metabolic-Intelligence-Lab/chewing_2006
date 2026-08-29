"""Stage 3 - build the per-subject master dataset (OFFICIAL-algorithm basis).

Signal features come from official.py (a validated 1:1 port of the lab pipeline
analisi_chewing.py; reproduces database_finale_completo.xlsx exactly). Snake_case
aliases are added so downstream modules (stats/clustering/swallowing) are stable.
Coordination metrics are derived from the official phases/events. Metadata comes
from database_finale_completo.xlsx.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import official
import coordination as coord
import swallow_morphology as morph

DB_FINAL = "database_finale_completo.xlsx"

# raw (un-normalised) swallow-morphology aggregates kept in the master. The
# within-channel %reference normalisation was found to REMOVE the muscle signal
# (absolute mylohyoid amplitude carries it), so normalised aggregates are kept
# only in the event-level CSV, not in the master.
MORPH_KEEP = ["sw_duration_s_mean", "sw_rise_time_s_mean", "sw_decay_time_s_mean",
              "sw_rise_decay_ratio_mean", "sw_fwhm_s_mean", "sw_rfd_mean",
              "sw_peak_mean", "sw_n_subpeaks_mean", "sw_piecemeal_frac",
              "sw_peak_cv", "sw_duration_s_cv"]

META_MAP = {
    "CHEWING_BIA - Peso (kg)": "weight_kg",
    "CHEWING_BIA - BMI": "bmi",
    "CHEWING_BIA - % Massa Grassa": "fat_pct",
    "CHEWING_BIA - Massa Magra%": "muscle_pct",
    "CHEWING_BIA - Grasso Viscerale %": "visceral_fat",
    "CHEWING_BIA - RM": "resting_metab",
    "CHEWING_HANDGRIP dx": "handgrip_dx",
    "CHEWING_HANDGRIP sx": "handgrip_sx",
    "CHEWING_Vo2_max": "vo2max",
    "QUESTIONARI_Inserisci la tua\xa0età (in anni):": "age",
    "QUESTIONARI_Inserisci il tuo genere": "sex",
    "QUESTIONARI_Inserisci la tua\xa0altezza\xa0(cm):": "height_cm",
    "QUESTIONARI_Sei un\xa0fumatore:": "smoker",
    "QUESTIONARI_Qual è la tua\xa0mano dominante?": "dominant_hand",
    "QUESTIONARI_Come valuteresti il\xa0livello di dolore\xa0durante la masticazione?": "chew_pain",
    "QUESTIONARI_Mastichi più frequentemente da un\xa0lato\xa0piuttosto che dall'altro?": "chew_side",
    "QUESTIONARI_Hai\xa0difficoltà a deglutire (disfagia)?": "dysphagia",
    "QUESTIONARI_Soffri o hai sofferto di\xa0reflusso gastroesofageo?": "reflux",
    "QUESTIONARI_Soffri o hai sofferto di\xa0cefalea?": "headache",
    "QUESTIONARI_Indossi\xa0byte\xa0notturni?": "night_byte",
    "QUESTIONARI_Porti\xa0protesi dentarie?": "dental_prosthesis",
    "QUESTIONARI_Presenti\xa0occlusioni dentali\xa0e/o\xa0assenza di denti\xa0che ti impediscono di masticare in maniera spontanea?": "occlusion_problem",
    "QUESTIONARI_Qual è la tua\xa0classe di occlusione? Consulta l'immagine allegata in alto per identificare la tua classe.": "occlusion_class",
    "QUESTIONARI_Soffri o hai sofferto della\xa0sindrome della lingua legata\xa0(frenulo della lingua corto)?": "tongue_tie",
    "QUESTIONARI_Hai mai sperimentato\xa0fascicolazioni\xa0(tremori o contrazioni muscolari involontarie)?": "fasciculations",
    "QUESTIONARI_Svolgi attività fisica regolare (2 o più volte a settimana)?": "phys_activity",
    "QUESTIONARI_Nell'ultimo mese, come valuti complessivamente la qualità del tuo sonno?": "sleep_quality",
    "QUESTIONARI_Hai riscontrato una diminuzione dell'appetito nell'ultimo mese?": "appetite_loss",
    "QUESTIONARI_Hai perso peso involontariamente?": "weight_loss",
}
NUMERIC_META = ["weight_kg", "bmi", "fat_pct", "muscle_pct", "visceral_fat",
                "resting_metab", "handgrip_dx", "handgrip_sx", "vo2max",
                "age", "height_cm"]


def _to_num(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if "," in s:
        parts = s.split(",")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s)
    except ValueError:
        return np.nan


def signal_table(raw_dir: str, res=None) -> pd.DataFrame:
    res = res or official.analyze_all(raw_dir)
    rows = []
    for rid, (summary, extra) in res.items():
        fs = extra["fs"]
        dur = len(extra["mast"]) / fs
        events = extra["events"]
        # swallow descriptors from official events
        ev_starts = sorted(s for s, _ in events)
        isi = float(np.diff(np.array(ev_starts) / fs).mean()) if len(ev_starts) > 1 else np.nan
        sw_dur = float(np.mean([(e - s) / fs for s, e in events])) if events else np.nan
        sw_peak = float(np.mean([extra["deg"][s:e].max() for s, e in events])) if events else np.nan
        extra["n_chews"] = summary["number of chews"]
        row = dict(summary)
        row["ID"] = int(rid)
        # snake_case aliases (chewing)
        row.update(dict(
            duration_s=dur,
            chew_phases=summary["number of chewing events"],
            chew_n=summary["number of chews"],
            chew_active_s=summary["chewing time_s"],
            chew_work=summary["work"],
            chew_work_rate=summary["work rate"],
            chew_tcyc_s=summary["Cycle Time"],
            chew_rate_hz=(summary["number of chews"] / dur) if dur else np.nan,
            # swallowing
            swallow_events=summary["number of deglutition events"],
            swallow_n=summary["number of deglutition"],
            swallow_active_s=summary["deglutition time_s"],
            swallow_work=summary["deglutition work"],
            swallow_work_rate=summary["deglutition work rate"],
            swallow_tcyc_s=summary["deglutition cycle time"],
            swallow_rate_per_min=(summary["number of deglutition"] / dur * 60.0) if dur else np.nan,
            swallow_dur_mean=sw_dur, swallow_peak_mean=sw_peak,
            inter_swallow_interval_s=isi,
            swallow_active_frac=(summary["deglutition time_s"] / dur) if dur else np.nan,
        ))
        row.update(coord.coordination_official(extra))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def metadata_table(db_path: str) -> pd.DataFrame:
    db = pd.read_excel(db_path)
    db = db.dropna(subset=["ID"]).copy()
    db["ID"] = db["ID"].astype(int)
    keep = ["ID"] + [c for c in META_MAP if c in db.columns]
    m = db[keep].rename(columns=META_MAP).copy()
    for c in NUMERIC_META:
        if c in m.columns:
            m[c] = m[c].map(_to_num)
    if "height_cm" in m.columns:
        m.loc[m["height_cm"] < 3, "height_cm"] *= 100
    if "vo2max" in m.columns:
        m.loc[m["vo2max"] > 100, "vo2max"] = np.nan   # inconsistent units -> drop
    if "handgrip_dx" in m.columns and "handgrip_sx" in m.columns:
        m["handgrip_max"] = m[["handgrip_dx", "handgrip_sx"]].max(axis=1)
        m["handgrip_mean"] = m[["handgrip_dx", "handgrip_sx"]].mean(axis=1)
    if {"weight_kg", "muscle_pct", "height_cm"} <= set(m.columns):
        h_m = m["height_cm"] / 100.0
        muscle_kg = m["weight_kg"] * m["muscle_pct"] / 100.0
        m["muscle_mass_kg"] = muscle_kg
        m["muscle_index"] = muscle_kg / (h_m ** 2)
    return m


def build_master(raw_dir: str, db_path: str) -> pd.DataFrame:
    res = official.analyze_all(raw_dir)            # computed once, reused
    sig = signal_table(raw_dir, res=res)
    # per-subject swallow-morphology aggregates (raw)
    events = morph.build_event_table(raw_dir, official_res=res)
    agg = morph.subject_aggregates(events)
    keep = ["ID"] + [c for c in MORPH_KEEP if c in agg.columns]
    sig = sig.merge(agg[keep], on="ID", how="left")
    meta = metadata_table(db_path)
    # inner join: keep only subjects present in the final official database
    df = sig.merge(meta, on="ID", how="inner")
    # minors are excluded from the dataset entirely (adults-only cohort)
    df = df[df["age"] >= 18].reset_index(drop=True)
    df["is_adult"] = True
    df["valid_swallow"] = df["swallow_n"] > 0
    return df


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    df = build_master(str(base / "dati_raw"), str(base / DB_FINAL))
    df.to_csv(out / "master_dataset.csv", index=False)
    print("master_dataset.csv:", df.shape)
    print("adults:", int(df.is_adult.sum()), "| valid_swallow:", int(df.valid_swallow.sum()),
          "| adults+valid:", int((df.is_adult & df.valid_swallow).sum()))
