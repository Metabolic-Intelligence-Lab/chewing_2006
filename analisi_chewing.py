# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.ndimage import median_filter


# =========================
# PATH E PARAMETRI
# =========================

BASE_PATH = r"D:\chewing\analisi dati post MF25\dati_raw"
OUTPUT_FOLDER = r"D:\chewing\analisi dati post MF25"
OUTPUT_XLSX = os.path.join(OUTPUT_FOLDER, "analisi_completa.xlsx")
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_FOLDER, "plots")

IDS_TO_ANALYZE = None  # None = analizza tutti i file .txt nella cartella
SAMPLING_RATE = 100
SHOW_PLOTS = False  # False = salva i plot senza aprirli a schermo


# =========================
# LETTURA DATI
# =========================

def load_and_preprocess_data(file_path):
    data_raw = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            if parts[0].strip().lower() == "data":
                continue
            try:
                values = parts[2].strip().split()
                if len(values) < 2:
                    continue
                data_raw.append([int(float(values[0])), int(float(values[1]))])
            except ValueError:
                continue
    return np.array(data_raw, dtype=float)


def adjust_signal(signal, sampling_rate=100, baseline_seconds=0.5):
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return signal
    n_offset = min(int(baseline_seconds * sampling_rate), len(signal))
    if n_offset < 1:
        n_offset = 1
    signal_adjusted = signal - np.mean(signal[:n_offset])
    signal_adjusted[signal_adjusted < 0] = 0
    return signal_adjusted


# =========================
# UTILITY
# =========================

def calculate_num_events(events):
    return len(events)


def calculate_total_events_time(events, sampling_rate=100):
    if len(events) == 0:
        return 0.0
    return sum(end - start for start, end in events) / sampling_rate


def estrai_id_da_nome_file(file_name):
    nome_senza_estensione = os.path.splitext(file_name)[0]
    primo_blocco = nome_senza_estensione.split("_")[0]
    if primo_blocco.isdigit():
        return f"{int(primo_blocco):03d}"
    match = re.search(r"\d+", primo_blocco)
    if match:
        return f"{int(match.group()):03d}"
    return None


def find_file_by_id(base_path, target_id):
    target_id = f"{int(target_id):03d}"
    for file_name in sorted(os.listdir(base_path)):
        if not file_name.lower().endswith(".txt"):
            continue
        file_id = estrai_id_da_nome_file(file_name)
        if file_id == target_id:
            return os.path.join(base_path, file_name), file_name
    return None, None


# =========================
# INIZIO ANALISI E SOGLIA
# =========================

def find_analysis_start(
    mastication_signal,
    mast_threshold,
    sampling_rate=100,
    min_sustained_s=1.0,
    mast_filter_size=51
):
    sm_mast = median_filter(mastication_signal, size=mast_filter_size)
    min_sustained = int(min_sustained_s * sampling_rate)
    above = sm_mast >= mast_threshold
    count = 0
    for i, is_above in enumerate(above):
        if is_above:
            count += 1
            if count >= min_sustained:
                return max(0, i - min_sustained + 1)
        else:
            count = 0
    peaks, _ = find_peaks(mastication_signal, height=10)
    return int(peaks[0]) if len(peaks) > 0 else 0


def compute_analysis_start_and_threshold(
    mastication_signal,
    sampling_rate=100,
    mast_filter_size=51,
    percentile=25,
    min_sustained_s=1.0,
    min_active=5,
    min_delay_after_start_s=5.0
):
    sm_mast = median_filter(mastication_signal, size=mast_filter_size)
    peaks, _ = find_peaks(mastication_signal, height=10)
    first_peak = int(peaks[0]) if len(peaks) > 0 else 0
    active_rough = sm_mast[first_peak:]
    active_rough = active_rough[active_rough > min_active]
    if len(active_rough) > 0:
        thresh_rough = float(np.percentile(active_rough, percentile))
    else:
        thresh_rough = 35.0
    analysis_start = find_analysis_start(
        mastication_signal,
        thresh_rough,
        sampling_rate=sampling_rate,
        min_sustained_s=min_sustained_s,
        mast_filter_size=mast_filter_size
    )
    active = sm_mast[analysis_start:]
    active = active[active > min_active]
    if len(active) > 0:
        mast_threshold = float(np.percentile(active, percentile))
    else:
        mast_threshold = thresh_rough
    deg_candidates_start = analysis_start + int(min_delay_after_start_s * sampling_rate)
    return analysis_start, deg_candidates_start, mast_threshold


# =========================
# FASI MASTICATORIE
# =========================

def identify_mastication_phases(
    mastication_signal_adjusted,
    mast_threshold,
    sampling_rate=100,
    mast_filter_size=51,
    min_pause_s=3.0,
    min_phase_duration_s=3.0
):
    mastication_signal_adjusted = np.asarray(mastication_signal_adjusted, dtype=float)
    if len(mastication_signal_adjusted) == 0:
        return []
    sm = median_filter(mastication_signal_adjusted, size=mast_filter_size)
    min_pause = int(min_pause_s * sampling_rate)
    min_phase = int(min_phase_duration_s * sampling_rate)
    above = sm >= mast_threshold
    phases = []
    in_phase = False
    phase_start = 0
    below_count = 0
    for i, is_above in enumerate(above):
        if is_above:
            if not in_phase:
                in_phase = True
                phase_start = i
            below_count = 0
        else:
            if in_phase:
                below_count += 1
                if below_count >= min_pause:
                    end = i - below_count
                    if (end - phase_start) >= min_phase:
                        phases.append((phase_start, end))
                    in_phase = False
                    below_count = 0
    if in_phase:
        end = len(mastication_signal_adjusted) - 1
        if (end - phase_start) >= min_phase:
            phases.append((phase_start, end))
    return phases


# =========================
# ZONE ESCLUSE E CONSENTITE
# =========================

def build_excluded_zones(
    deg_candidates_start,
    mastication_phases,
    sampling_rate=100,
    phase_start_buffer_s=3.0
):
    excluded = []
    buf = int(phase_start_buffer_s * sampling_rate)
    excluded.append((0, deg_candidates_start))
    for phase_start, phase_end in mastication_phases:
        excl_end = min(phase_start + buf, phase_end)
        if excl_end > phase_start:
            excluded.append((phase_start, excl_end))
    return excluded


def is_in_excluded_zone(sample_idx, excluded_zones):
    for start, end in excluded_zones:
        if start <= sample_idx < end:
            return True
    return False


def overlaps_excluded_zone(start, end, excluded_zones):
    for zone_start, zone_end in excluded_zones:
        if start < zone_end and end > zone_start:
            return True
    return False


def build_deglutition_allowed_zones(
    mastication_phases,
    signal_length,
    sampling_rate=100,
    pre_phase_end_s=1.0,
    post_phase_end_s=8.0
):
    allowed = []
    pre = int(pre_phase_end_s * sampling_rate)
    post = int(post_phase_end_s * sampling_rate)
    for phase_start, phase_end in mastication_phases:
        start = max(0, phase_end - pre)
        end = min(signal_length - 1, phase_end + post)
        if end > start:
            allowed.append((start, end))
    return allowed


def is_in_allowed_zone(sample_idx, allowed_zones):
    if len(allowed_zones) == 0:
        return True
    for start, end in allowed_zones:
        if start <= sample_idx <= end:
            return True
    return False


def overlaps_allowed_zone(start, end, allowed_zones):
    if len(allowed_zones) == 0:
        return True
    for zone_start, zone_end in allowed_zones:
        if start < zone_end and end > zone_start:
            return True
    return False


# =========================
# RILEVAZIONE DEGLUTIZIONI
# =========================

def identify_deglutition_events(
    deglutition_signal_adjusted,
    mastication_signal_adjusted,
    mastication_phases=None,
    sampling_rate=100,
    env_filter_size=151,
    env_peak_height=8,
    env_prominence=5,
    env_peak_distance=80,
    border_fraction=0.10,
    mast_filter_size=51,
    adaptive_percentile=25,
    min_sustained_s=1.0,
    min_delay_after_start_s=5.0,
    phase_start_buffer_s=3.0,
    use_phase_end_windows=True,
    pre_phase_end_s=1.0,
    post_phase_end_s=8.0,
    mast_relax_factor=1.50,
    max_derivative_sign_changes_per_s=35.0,
    max_derivative_activity_ratio=4.00,
    min_deg_amplitude=10,
    min_duration_s=0.15,
    max_duration_s=5.0,
    fine_filter_size=7
):
    deglutition_signal_adjusted = np.asarray(deglutition_signal_adjusted, dtype=float)
    mastication_signal_adjusted = np.asarray(mastication_signal_adjusted, dtype=float)

    analysis_start, deg_candidates_start, mast_threshold = compute_analysis_start_and_threshold(
        mastication_signal_adjusted,
        sampling_rate=sampling_rate,
        mast_filter_size=mast_filter_size,
        percentile=adaptive_percentile,
        min_sustained_s=min_sustained_s,
        min_delay_after_start_s=min_delay_after_start_s
    )

    deg_envelope = median_filter(deglutition_signal_adjusted, size=env_filter_size)
    sm_mast = median_filter(mastication_signal_adjusted, size=mast_filter_size)
    deg_fine = median_filter(deglutition_signal_adjusted, size=fine_filter_size)

    excluded_zones = build_excluded_zones(
        deg_candidates_start,
        mastication_phases if mastication_phases is not None else [],
        sampling_rate=sampling_rate,
        phase_start_buffer_s=phase_start_buffer_s
    )

    if use_phase_end_windows:
        allowed_zones = build_deglutition_allowed_zones(
            mastication_phases if mastication_phases is not None else [],
            signal_length=len(deglutition_signal_adjusted),
            sampling_rate=sampling_rate,
            pre_phase_end_s=pre_phase_end_s,
            post_phase_end_s=post_phase_end_s
        )
    else:
        allowed_zones = []

    all_peaks, all_props = find_peaks(
        deg_envelope,
        height=env_peak_height,
        distance=env_peak_distance,
        prominence=env_prominence
    )

    if len(all_peaks) == 0:
        return [], mast_threshold, analysis_start, deg_candidates_start

    valid_mask = all_peaks >= deg_candidates_start
    env_peaks = all_peaks[valid_mask]
    prominences = all_props["prominences"][valid_mask]

    if len(env_peaks) == 0:
        return [], mast_threshold, analysis_start, deg_candidates_start

    env_valleys = []
    for i in range(len(env_peaks) - 1):
        seg = deg_envelope[env_peaks[i]:env_peaks[i + 1]]
        valley = int(np.argmin(seg)) + env_peaks[i]
        env_valleys.append(valley)

    deglutition_events = []

    for pk_idx, pk in enumerate(env_peaks):
        if pk_idx == 0:
            left_bound = max(deg_candidates_start, pk - int(2.0 * sampling_rate))
        else:
            left_bound = env_valleys[pk_idx - 1]

        if pk_idx == len(env_peaks) - 1:
            right_bound = min(len(deglutition_signal_adjusted) - 1, pk + int(2.0 * sampling_rate))
        else:
            right_bound = env_valleys[pk_idx]

        half_amp = deg_envelope[pk] * border_fraction

        left = pk
        while left > left_bound and deg_envelope[left] > half_amp:
            left -= 1

        right = pk
        while right < right_bound and deg_envelope[right] > half_amp:
            right += 1

        if right - left < int(min_duration_s * sampling_rate):
            left = max(deg_candidates_start, pk - int(0.3 * sampling_rate))
            right = min(len(deglutition_signal_adjusted) - 1, pk + int(0.5 * sampling_rate))

        dur = (right - left) / sampling_rate

        max_deg_amp = 0.0
        if right > left:
            max_deg_amp = float(deglutition_signal_adjusted[left:right].max())

        if is_in_excluded_zone(pk, excluded_zones) or overlaps_excluded_zone(left, right, excluded_zones):
            continue

        if use_phase_end_windows and not (
            is_in_allowed_zone(pk, allowed_zones)
            or overlaps_allowed_zone(left, right, allowed_zones)
        ):
            continue

        if not (min_duration_s <= dur <= max_duration_s):
            continue

        amp_ok = max_deg_amp >= min_deg_amplitude

        seg = deg_fine[left:right]
        d_seg = np.diff(seg)

        if len(d_seg) == 0:
            continue

        sc = np.sign(d_seg)
        sc = sc[sc != 0]

        if len(sc) > 1:
            sc_per_s = float(np.sum(sc[1:] != sc[:-1]) / dur)
        else:
            sc_per_s = 0.0

        mean_amp_seg = float(np.mean(seg))
        dar = float(np.mean(np.abs(d_seg)) / mean_amp_seg) if mean_amp_seg > 0 else 0.0

        deriv_ok = (
            sc_per_s <= max_derivative_sign_changes_per_s
            and dar <= max_derivative_activity_ratio
        )

        local_mast = float(sm_mast[left:right].mean())
        mast_ok = (
            local_mast < mast_threshold * mast_relax_factor
            or is_in_allowed_zone(pk, allowed_zones)
        )

        if amp_ok and deriv_ok and mast_ok:
            deglutition_events.append((left, right))

    return deglutition_events, mast_threshold, analysis_start, deg_candidates_start


# =========================
# METRICHE
# =========================

def segment_active_parts(signal, threshold=1, min_len=6, min_area=100, sample_period=0.01):
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return [], [], 0.0, 0.0
    segments = []
    start = None
    for i, value in enumerate(signal):
        if value >= threshold and start is None:
            start = i
        elif value < threshold and start is not None:
            segments.append(signal[start:i])
            start = None
    if start is not None:
        segments.append(signal[start:])
    valid = [
        np.array(seg, dtype=float)
        for seg in segments
        if len(seg) > min_len and np.sum(seg) > min_area
    ]
    work = 0.0
    total_time = 0.0
    segment_times = []
    for seg in valid:
        n = len(seg)
        if n <= 1:
            continue
        mean_volt = (np.sum(seg) / n) * 0.001
        time_s = sample_period * (n - 1)
        work += mean_volt * time_s
        total_time += time_s
        segment_times.append(time_s)
    return valid, segment_times, work, total_time


def compute_metrics_on_events(signal, events, sampling_rate=100, threshold=1, min_len=6, min_area=100):
    signal = np.asarray(signal, dtype=float)
    sample_period = 1.0 / sampling_rate
    total_count = 0
    total_time_s = 0.0
    total_work = 0.0
    for start, end in events:
        segments, segment_times, work, active_time = segment_active_parts(
            signal[start:end],
            threshold=threshold,
            min_len=min_len,
            min_area=min_area,
            sample_period=sample_period
        )
        count = len(segments)
        total_count += count
        total_time_s += active_time
        total_work += work
    summary = {
        "count_total": int(total_count),
        "active_time_s": round(total_time_s, 6),
        "mean_cycle_time_s": round(total_time_s / total_count, 6) if total_count > 0 else 0.0,
        "work_total": round(total_work, 6),
        "work_rate": round(total_work / total_time_s, 6) if total_time_s > 0 else 0.0,
    }
    return summary


# =========================
# COLORI FASI
# =========================

PHASE_COLORS = [
    "#a8d8ea", "#ffd6a5", "#caffbf", "#fdffb6",
    "#c8b6ff", "#ffb3c6", "#b9fbc0", "#ffc8dd",
]


# =========================
# PLOT 1
# =========================

def plot_full_signal_with_phases(
    mastication_signal,
    deglutition_signal,
    mastication_phases,
    deglutition_events,
    deg_candidates_start,
    mast_threshold,
    file_id,
    sampling_rate=100,
    output_dir=None,
    show_plot=False
):
    time_axis = np.arange(len(mastication_signal)) / sampling_rate
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(time_axis, mastication_signal, label="Canale masticazione", color="blue", linewidth=1, zorder=3)
    ax.plot(time_axis, deglutition_signal, label="Canale deglutizione", color="orange", alpha=0.75, linewidth=1, zorder=3)

    y_top = max(
        float(np.max(mastication_signal)) if len(mastication_signal) > 0 else 1,
        float(np.max(deglutition_signal)) if len(deglutition_signal) > 0 else 1
    ) * 1.08
    ax.set_ylim(bottom=-20, top=y_top)

    phase_patches = []
    for idx, (start, end) in enumerate(mastication_phases):
        color = PHASE_COLORS[idx % len(PHASE_COLORS)]
        dur = (end - start) / sampling_rate
        t_start = start / sampling_rate
        t_end = end / sampling_rate
        t_mid = (t_start + t_end) / 2
        ax.axvspan(t_start, t_end, color=color, alpha=0.55, zorder=1)
        ax.text(t_mid, y_top * 0.98, f"Fase {idx + 1}\n{dur:.0f}s",
                ha="center", va="top", fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.7, ec="none"))
        phase_patches.append(mpatches.Patch(color=color, alpha=0.55,
                                            label=f"Fase {idx + 1}: {t_start:.1f}-{t_end:.1f}s ({dur:.0f}s)"))

    for idx, (start, end) in enumerate(deglutition_events):
        ax.axvspan(start / sampling_rate, end / sampling_rate, color="red", alpha=0.40, zorder=2,
                   label="Evento deglutizione" if idx == 0 else None)

    ax.axvline(x=deg_candidates_start / sampling_rate, color="black", linestyle="--", linewidth=1,
               label="Inizio ricerca deglutizioni")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + phase_patches, fontsize=7, loc="upper right")
    ax.set_title(f"ID {file_id} — Fasi masticatorie e deglutizioni")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Ampiezza corretta")
    ax.grid(True)
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ID_{file_id}_fasi_masticatorie_deglutizioni.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Plot salvato: {output_path}")

    if show_plot:
        plt.show(block=True)

    plt.close(fig)


# =========================
# ANALISI SINGOLO ID
# =========================

def analyze_single_id(
    file_path,
    file_id,
    sampling_rate=100,
    output_plots_dir=None,
    show_plots=False
):
    data_array = load_and_preprocess_data(file_path)

    if data_array.size == 0:
        return None

    mastication_signal = adjust_signal(data_array[:, 0], sampling_rate=sampling_rate)
    deglutition_signal = adjust_signal(data_array[:, 1], sampling_rate=sampling_rate)

    analysis_start, deg_candidates_start, mast_threshold = compute_analysis_start_and_threshold(
        mastication_signal,
        sampling_rate=sampling_rate,
        mast_filter_size=51,
        percentile=25,
        min_sustained_s=1.0,
        min_delay_after_start_s=5.0
    )

    mastication_phases = identify_mastication_phases(
        mastication_signal,
        mast_threshold=mast_threshold,
        sampling_rate=sampling_rate,
        mast_filter_size=51,
        min_pause_s=3.0,
        min_phase_duration_s=3.0
    )

    deglutition_events, mast_threshold, analysis_start, deg_candidates_start = identify_deglutition_events(
        deglutition_signal,
        mastication_signal,
        mastication_phases=mastication_phases,
        sampling_rate=sampling_rate,
        env_filter_size=151,
        env_peak_height=8,
        env_prominence=5,
        env_peak_distance=80,
        border_fraction=0.10,
        mast_filter_size=51,
        adaptive_percentile=25,
        min_sustained_s=1.0,
        min_delay_after_start_s=5.0,
        phase_start_buffer_s=3.0,
        use_phase_end_windows=True,
        pre_phase_end_s=1.0,
        post_phase_end_s=8.0,
        mast_relax_factor=1.50,
        max_derivative_sign_changes_per_s=35.0,
        max_derivative_activity_ratio=4.00,
        min_deg_amplitude=10,
        min_duration_s=0.15,
        max_duration_s=5.0,
        fine_filter_size=7
    )

    mastication_summary = compute_metrics_on_events(mastication_signal, mastication_phases, sampling_rate=sampling_rate)
    deglutition_summary = compute_metrics_on_events(deglutition_signal, deglutition_events, sampling_rate=sampling_rate)

    plot_full_signal_with_phases(
        mastication_signal, deglutition_signal, mastication_phases, deglutition_events,
        deg_candidates_start=deg_candidates_start, mast_threshold=mast_threshold,
        file_id=file_id, sampling_rate=sampling_rate, output_dir=output_plots_dir, show_plot=show_plots
    )

    chewing_time_s = round(calculate_total_events_time(mastication_phases, sampling_rate=sampling_rate), 6)
    deglutition_time_s = round(calculate_total_events_time(deglutition_events, sampling_rate=sampling_rate), 6)

    summary = {
        "ID": file_id,
        "number of chewing events": calculate_num_events(mastication_phases),
        "chewing time_s": chewing_time_s,
        "chewing time_min": round(chewing_time_s / 60.0, 6),
        "number of chews": mastication_summary["count_total"],
        "work": mastication_summary["work_total"],
        "work rate": mastication_summary["work_rate"],
        "Cycle Time": mastication_summary["mean_cycle_time_s"],
        "number of deglutition events": calculate_num_events(deglutition_events),
        "deglutition time_s": deglutition_time_s,
        "deglutition time_min": round(deglutition_time_s / 60.0, 6),
        "number of deglutition": deglutition_summary["count_total"],
    }

    return summary


# =========================
# ANALISI BATCH
# =========================

def run_analysis(
    ids_to_analyze=IDS_TO_ANALYZE,
    base_path=BASE_PATH,
    output_xlsx=OUTPUT_XLSX,
    output_plots_dir=OUTPUT_PLOTS_DIR,
    sampling_rate=100,
    show_plots=False
):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Cartella dati_raw non trovata: {base_path}")

    if ids_to_analyze is None:
        ids_to_analyze = []
        for file_name in sorted(os.listdir(base_path)):
            if not file_name.lower().endswith(".txt"):
                continue
            file_id = estrai_id_da_nome_file(file_name)
            if file_id is not None:
                ids_to_analyze.append(file_id)
        print(f"[INFO] Trovati {len(ids_to_analyze)} file: {ids_to_analyze}")

    summary_rows = []
    errors = []

    os.makedirs(output_plots_dir, exist_ok=True)

    for raw_id in ids_to_analyze:
        target_id = f"{int(raw_id):03d}"
        file_path, file_name = find_file_by_id(base_path, target_id)

        if file_path is None:
            print(f"[WARN] Nessun file trovato per ID {target_id}")
            errors.append({"ID": target_id, "errore": "File non trovato"})
            continue

        print(f"\n[INFO] Analisi ID {target_id} -> {file_name}")

        try:
            summary = analyze_single_id(
                file_path=file_path,
                file_id=target_id,
                sampling_rate=sampling_rate,
                output_plots_dir=output_plots_dir,
                show_plots=show_plots
            )

            if summary is None:
                print(f"[WARN] File vuoto/non valido: {file_name}")
                errors.append({"ID": target_id, "errore": "File vuoto o non valido"})
                continue

            summary_rows.append(summary)

        except Exception as e:
            print(f"[ERROR] ID {target_id}: {e}")
            errors.append({"ID": target_id, "errore": str(e)})

    if len(summary_rows) == 0:
        print("[ATTENZIONE] Nessun risultato da salvare.")
        return

    df_summary = pd.DataFrame(summary_rows)
    df_errors = pd.DataFrame(errors)

    output_folder = os.path.dirname(output_xlsx)
    os.makedirs(output_folder, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="summary_ID")
        if len(df_errors) > 0:
            df_errors.to_excel(writer, index=False, sheet_name="errori")

    print("\n========== ANALISI COMPLETATA ==========")
    print(f"[OK] Excel salvato: {output_xlsx}")
    print(f"[OK] Plot salvati in: {output_plots_dir}")
    print(f"[OK] ID analizzati: {len(df_summary)}")
    if len(df_errors) > 0:
        print(f"[ATTENZIONE] Errori: {len(df_errors)}")
        print(df_errors)
    print("========================================")


# =========================
# ESECUZIONE
# =========================

if __name__ == "__main__":
    run_analysis(
        ids_to_analyze=IDS_TO_ANALYZE,
        base_path=BASE_PATH,
        output_xlsx=OUTPUT_XLSX,
        output_plots_dir=OUTPUT_PLOTS_DIR,
        sampling_rate=SAMPLING_RATE,
        show_plots=SHOW_PLOTS
    )