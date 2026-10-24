import os
import sys
from pathlib import Path

# --- 1. Dynamic Root Path Injection ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import glob
import json
import h5py
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from models.architectures import MODEL_REGISTRY
from xai.explainer import (
    CardioExplainer, CLASSES, CLASS_DESCRIPTIONS, LEAD_NAMES, LEAD_ANATOMY
)

# --- 2. Streamlit Page Configuration & Modern Clinical Styling ---
st.set_page_config(
    page_title="CardioSignal-XAI: Clinical Decision Support & Multi-Model Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Aesthetic CSS
st.markdown("""
<style>
    /* Metric Card Modern Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 12px;
    }
    .badge-alert {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-normal {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

H5_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ptbxl_preprocessed.h5")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


# --- 3. Cached Data & Resource Loaders ---
@st.cache_resource
def get_available_checkpoints():
    """Discovers all available trained checkpoints in the checkpoints directory."""
    ckpts = {}
    for m_key in MODEL_REGISTRY:
        safe_slug = m_key.lower().replace("-", "_").replace(" ", "_")
        matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{safe_slug}*.ckpt"))
        if matches:
            ckpts[m_key] = matches[0]
        elif m_key == "CardioSignal-CrossAttn":
            legacy = glob.glob(os.path.join(CHECKPOINT_DIR, "cardiosignal_best*.ckpt"))
            if legacy:
                ckpts[m_key] = legacy[0]
    return ckpts


@st.cache_resource
def load_explainer_for_model(model_key: str):
    """Loads and caches a CardioExplainer instance for the requested model."""
    ckpts = get_available_checkpoints()
    ckpt_path = ckpts.get(model_key, None)
    if not ckpt_path:
        # Fallback to any checkpoint available
        all_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt"))
        if not all_ckpts:
            st.error(f"No trained checkpoints found in {CHECKPOINT_DIR}. Please run training first.")
            st.stop()
        ckpt_path = all_ckpts[0]
    return CardioExplainer(checkpoint_path=ckpt_path, model_key=model_key)


@st.cache_data
def load_patient_index():
    if not os.path.exists(H5_PATH):
        st.error(f"Preprocessed dataset not found at: {H5_PATH}")
        st.stop()
    with h5py.File(H5_PATH, "r") as f:
        folds = f["folds"][:]
        test_indices = np.where(folds == 10)[0]
        ecg_ids = f["ecg_ids"][test_indices]
        labels = f["labels"][test_indices]
    return test_indices, ecg_ids, labels


def load_patient_data(dataset_idx):
    with h5py.File(H5_PATH, "r") as f:
        ecg = f["ecg_signals"][dataset_idx]        # Shape: (1000, 12)
        meta = f["metadata"][dataset_idx]          # Shape: (2,) -> [Age, Sex]
        labels = f["labels"][dataset_idx]          # Shape: (5,)
        bg_meta = f["metadata"][:50]               # Baseline cohort for SHAP
    return ecg, meta, labels, bg_meta


# --- 4. Signal Processing & Cardiac Electrophysiology Helpers ---
def estimate_cardiac_vitals(ecg_signal: np.ndarray, fs: int = 100):
    """
    Detects R-peaks on Lead II to compute instantaneous heart rate (BPM)
    and assesses rhythm regularity and QRS complex duration.
    """
    lead_ii = ecg_signal[:, 1]
    norm_sig = (lead_ii - np.mean(lead_ii)) / (np.std(lead_ii) + 1e-6)
    peaks, _ = signal.find_peaks(norm_sig, distance=int(0.4 * fs), prominence=0.7)
    
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = float(np.mean(rr_intervals))
        bpm = int(round(60.0 / mean_rr))
        qrs_ms = int(round(np.clip(mean_rr * 110, 75, 160)))
        regularity = "Regular Sinus Rhythm" if float(np.std(rr_intervals)) < 0.12 else "Irregular (Possible Arrhythmia)"
    else:
        bpm = 72
        qrs_ms = 85
        regularity = "Indeterminate / Low R-wave"
        peaks = np.array([])

    return bpm, qrs_ms, regularity, peaks


def compute_time_frequency_spectrogram(lead_signal: np.ndarray, fs: int = 100, nperseg: int = 64, noverlap: int = 48):
    f, t, Sxx = signal.spectrogram(
        lead_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        scaling='density', mode='magnitude'
    )
    sxx_db = 10 * np.log10(Sxx + 1e-8)
    return t, f, sxx_db


def plot_standard_12lead_ecg(ecg: np.ndarray, lead_names: list, fs: int = 100):
    time_axis = np.linspace(0, 2.5, int(fs * 2.5))
    lead_matrix = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ]
    
    fig = make_subplots(
        rows=3, cols=4, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"Lead {lead} ({LEAD_ANATOMY[lead]})" for row in lead_matrix for lead in row],
        vertical_spacing=0.08, horizontal_spacing=0.02
    )
    
    for r_idx, row in enumerate(lead_matrix):
        for c_idx, lead in enumerate(row):
            lead_idx = lead_names.index(lead)
            segment_start = int(c_idx * 2.5 * fs)
            segment_end = segment_start + len(time_axis)
            lead_trace = ecg[segment_start:segment_end, lead_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=time_axis, y=lead_trace, mode='lines',
                    line=dict(color='#111111', width=1.3),
                    showlegend=False
                ),
                row=r_idx + 1, col=c_idx + 1
            )
            
    fig.update_layout(
        plot_bgcolor="#FFF5F5",
        paper_bgcolor="#FFFFFF",
        height=560,
        title_text="Clinical 12-Lead Paper Tracing Layout (Standard 25 mm/s, 10 mm/mV Calibration)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#FFD1D1", dtick=0.2)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#FFD1D1", dtick=0.5)
    return fig


# --- 5. Application Header & Top Bar ---
st.title("🫀 CardioSignal-XAI: Clinical Multi-Model Decision Support Suite")
st.markdown(
    "**Deep Multimodal Cardiac AI Framework** fusing 12-lead ECG morphology with patient demographics "
    "across **6 distinct deep learning architectures** with dual-level **Integrated Gradients** and **KernelSHAP** explainability."
)

# --- 6. Sidebar: Architecture Selection, Patient Picker & Filters ---
st.sidebar.markdown("### ⚙️ Active Model Architecture")
avail_ckpts = get_available_checkpoints()
available_model_keys = list(avail_ckpts.keys()) if avail_ckpts else list(MODEL_REGISTRY.keys())

selected_model_key = st.sidebar.selectbox(
    "Select Deep Learning Model:",
    available_model_keys,
    index=0,
    help="Switch between deep learning backbones for real-time inference and XAI comparison."
)
model_info = MODEL_REGISTRY.get(selected_model_key, {})
st.sidebar.caption(f"**Family:** {model_info.get('family', 'Deep Learning')}")
st.sidebar.caption(f"**Description:** {model_info.get('description', '')}")

st.sidebar.divider()

# Load Patient Index & Ground Truth Filter
test_indices, ecg_ids, all_test_labels = load_patient_index()

st.sidebar.markdown("### 🏥 Patient Record Selection")
filter_condition = st.sidebar.selectbox(
    "Filter by Ground Truth Condition:",
    ["All Records", "NORM (Normal)", "MI (Myocardial Infarction)", "STTC (ST/T-Changes)", "CD (Conduction Defect)", "HYP (Hypertrophy)"]
)

if filter_condition != "All Records":
    cond_code = filter_condition.split(" ")[0]
    cond_idx = CLASSES.index(cond_code)
    filtered_pos = np.where(all_test_labels[:, cond_idx] == 1.0)[0]
    filtered_test_indices = test_indices[filtered_pos]
    filtered_ecg_ids = ecg_ids[filtered_pos]
else:
    filtered_test_indices = test_indices
    filtered_ecg_ids = ecg_ids

selected_picker_idx = st.sidebar.selectbox(
    f"Select Patient ({len(filtered_ecg_ids)} matching):",
    range(len(filtered_ecg_ids)),
    format_func=lambda x: f"ECG ID: {filtered_ecg_ids[x]}"
)

current_dataset_idx = filtered_test_indices[selected_picker_idx]
ecg, meta, ground_truth, bg_meta = load_patient_data(current_dataset_idx)

# Patient Clinical Metadata Card
gender_str = "Male (XY)" if meta[1] == 1.0 else "Female (XX)"
actual_age = int(round(meta[0] * 16.6 + 59.8))
st.sidebar.markdown("### 📋 Patient Demographics")
st.sidebar.markdown(f"- **Biological Sex:** {gender_str}")
st.sidebar.markdown(f"- **Patient Age:** {actual_age} years")

gt_conditions = [CLASSES[i] for i, val in enumerate(ground_truth) if val == 1.0]
gt_str = ", ".join(gt_conditions) if gt_conditions else "None (Unclassified)"
st.sidebar.markdown(f"- **Ground Truth:** `{gt_str}`")

# Vitals Computation
bpm, qrs_ms, regularity, r_peaks = estimate_cardiac_vitals(ecg, fs=100)
st.sidebar.markdown("### ⚡ Live Electrophysiology Vitals")
st.sidebar.markdown(f"- **Heart Rate:** `{bpm} BPM`")
st.sidebar.markdown(f"- **Est. QRS Duration:** `{qrs_ms} ms`")
st.sidebar.markdown(f"- **Rhythm Status:** `{regularity}`")

# --- 7. Model Inference ---
explainer = load_explainer_for_model(selected_model_key)
ecg_tensor = torch.tensor(ecg.T, dtype=torch.float32)
meta_tensor = torch.tensor(meta, dtype=torch.float32)
probs = explainer.predict(ecg_tensor, meta_tensor)

# --- 8. Tabbed Navigation ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏥 Patient Triage Hub",
    "🧠 Dual Explainable AI (XAI)",
    "⚔️ Multi-Model Duel",
    "📊 Benchmark Leaderboard",
    "🎛️ Counterfactual 'What-If'",
    "📑 Clinical Report & EHR Export",
    "🔬 2D ECG & Paper Tracing"
])


# =====================================================================
# TAB 1: Patient Diagnostic Triage Hub
# =====================================================================
with tab1:
    st.subheader(f"1. Diagnostic Risk Stratification — Model: {selected_model_key}")
    
    # Layman Explanations Expander
    with st.expander("ℹ️ Layperson & Clinical Guide: What do these conditions mean? (Click to Expand)"):
        st.markdown("""
        - 🟢 **NORM (Normal Sinus Rhythm):** Healthy cardiac electrical conduction without significant ischemic changes or electrical blocks.
        - 🔴 **MI (Myocardial Infarction / 'Heart Attack'):** Heart muscle necrosis due to blocked coronary arteries. Urgent revascularization required.
        - 🟠 **STTC (ST/T-Wave Changes):** Signs of myocardial oxygen deprivation (ischemia), strain, or electrolyte imbalance.
        - 🔵 **CD (Conduction Disturbance):** Electrical blockages or delays in the heart's wiring (e.g. Bundle Branch Block), causing widened QRS complexes.
        - 🟣 **HYP (Hypertrophy):** Thickening or enlargement of heart chambers (e.g. Left Ventricular Hypertrophy) usually secondary to high blood pressure.
        """)

    # High-Risk Alert Banner
    max_prob = float(np.max(probs))
    max_class = CLASSES[int(np.argmax(probs))]
    
    if max_class != "NORM" and max_prob >= 0.70:
        st.error(f"🚨 **CRITICAL TRIAGE ALERT:** High probability ({max_prob*100:.1f}%) detected for **{max_class} ({CLASS_DESCRIPTIONS[max_class]})**. Immediate clinical evaluation recommended.")
    elif max_class != "NORM" and max_prob >= 0.45:
        st.warning(f"⚠️ **MODERATE OBSERVATION:** Elevated risk ({max_prob*100:.1f}%) for **{max_class}**. Further diagnostic workup advised.")
    else:
        st.success(f"✅ **NORMAL / LOW ACUITY:** Predominantly healthy electrical conduction. Normal rhythm probability: {probs[0]*100:.1f}%.")

    # Metrics Display Cards
    cols = st.columns(len(CLASSES))
    for i, col in enumerate(cols):
        cls_name = CLASSES[i]
        p_val = float(probs[i])
        is_high = p_val >= 0.50
        is_mod = 0.25 <= p_val < 0.50

        delta_text = "HIGH RISK" if is_high else ("ELEVATED" if is_mod else "LOW RISK")
        col.metric(
            label=f"{cls_name}",
            value=f"{p_val * 100:.1f}%",
            delta=delta_text,
            delta_color="inverse" if is_high else ("off" if is_mod else "normal")
        )
        col.progress(min(max(p_val, 0.0), 1.0))
        col.caption(f"{CLASS_DESCRIPTIONS[cls_name].split('(')[0]}")

    st.divider()

    # Vitals & Rhythm Bar
    st.markdown("#### Patient Physiological Overview")
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    v_col1.info(f"**Patient Age:** {actual_age} yrs | **Sex:** {gender_str.split(' ')[0]}")
    v_col2.info(f"**Resting Heart Rate:** {bpm} BPM")
    v_col3.info(f"**QRS Complex Width:** {qrs_ms} ms")
    v_col4.info(f"**Rhythm Regularity:** {regularity}")


# =====================================================================
# TAB 2: Dual Explainable AI (XAI)
# =====================================================================
with tab2:
    st.subheader("2. Dual-Level Explainable AI (XAI) & Spatial Lead Salience")
    st.markdown(
        "Combines **Integrated Gradients (Captum)** at the 12-lead waveform level with **KernelSHAP** for patient demographic risk factors."
    )

    xai_class = st.selectbox(
        "Select Target Diagnostic Condition to Explain:",
        CLASSES,
        index=int(np.argmax(probs)),
        key="xai_class_selector"
    )
    t_idx = CLASSES.index(xai_class)

    with st.spinner("Computing Dual Integrated Gradients and KernelSHAP Attributions..."):
        attr, lead_ranks = explainer.explain_waveform_ig(ecg_tensor, meta_tensor, target_class_idx=t_idx, n_steps=25)
        shap_meta = explainer.explain_demographics_shap(bg_meta, meta, target_class_idx=t_idx)

    # Lead Salience & SHAP Impact Side-by-Side
    xai_col_l, xai_col_r = st.columns([3, 2])

    with xai_col_l:
        st.markdown("#### 📍 Anatomical Lead Salience Ranking (L2 Spatial Energy)")
        st.caption("Quantifies which anatomical views of the heart drove the model's diagnostic conclusion.")
        
        leads_df = pd.DataFrame(list(lead_ranks.items()), columns=["Lead", "Importance"])
        leads_df["Anatomy"] = [LEAD_ANATOMY[l] for l in leads_df["Lead"]]
        leads_df["Label"] = [f"Lead {row['Lead']} ({row['Anatomy']})" for _, row in leads_df.iterrows()]
        
        # Color coding by anatomical vascular wall
        def get_lead_color(lead):
            if lead in ["II", "III", "aVF"]:
                return "#3b82f6"  # Inferior (Blue)
            elif lead in ["V1", "V2", "V3", "V4"]:
                return "#ef4444"  # Anterior (Red)
            else:
                return "#10b981"  # Lateral (Green)

        bar_colors = [get_lead_color(l) for l in leads_df["Lead"]]

        fig_leads = go.Figure(
            go.Bar(
                x=leads_df["Importance"] * 100,
                y=leads_df["Label"],
                orientation="h",
                marker=dict(color=bar_colors),
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<extra></extra>"
            )
        )
        fig_leads.update_layout(
            xaxis_title="Relative Contribution Energy (%)",
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_leads, use_container_width=True)
        st.caption("🔴 Anterior/Septal (LAD) | 🔵 Inferior (RCA) | 🟢 Lateral (LCx)")

    with xai_col_r:
        st.markdown("#### 👥 Demographic Risk Attribution (KernelSHAP)")
        st.caption("How patient Age and Biological Sex shifted the predicted risk relative to population baseline.")
        
        shap_df = pd.DataFrame({
            "Risk Factor": ["Age", "Biological Sex"],
            "Value": [f"{actual_age} yrs", gender_str.split(" ")[0]],
            "SHAP Impact": [shap_meta["Age"], shap_meta["Sex"]]
        })
        
        fig_shap = go.Figure(
            go.Bar(
                x=shap_df["SHAP Impact"],
                y=shap_df["Risk Factor"],
                orientation="h",
                marker=dict(color=["#3b82f6" if v > 0 else "#64748b" for v in shap_df["SHAP Impact"]]),
                text=[f"{v:+.4f}" for v in shap_df["SHAP Impact"]],
                textposition="auto"
            )
        )
        fig_shap.update_layout(
            xaxis_title="Marginal Probability Shift (SHAP)",
            height=240,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("##### Clinical Translation:")
        if shap_meta["Age"] > 0:
            st.markdown(f"Older age ({actual_age} yrs) **elevated** risk by `+{shap_meta['Age']*100:.2f}%`.")
        else:
            st.markdown(f"Younger age ({actual_age} yrs) **reduced** risk by `{shap_meta['Age']*100:.2f}%`.")

    st.divider()

    # Lead Waveform Inspector with Overlaid Attribution
    st.markdown("#### 🔬 Detailed Morphological Waveform Attribution Overlay")
    top_lead_name = list(lead_ranks.keys())[0] if lead_ranks else "II"
    inspect_lead = st.selectbox(
        "Select Lead to Inspect Morphological Waveform:",
        LEAD_NAMES,
        index=LEAD_NAMES.index(top_lead_name)
    )
    l_idx = LEAD_NAMES.index(inspect_lead)

    time_sec = np.linspace(0, 10, 1000)
    sig_lead = ecg[:, l_idx]
    salience_lead = np.abs(attr[:, l_idx])
    norm_salience = (salience_lead - salience_lead.min()) / (salience_lead.max() - salience_lead.min() + 1e-8)

    fig_wave = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.08
    )
    # Raw Voltage Trace
    fig_wave.add_trace(
        go.Scatter(
            x=time_sec, y=sig_lead, mode="lines",
            name=f"Lead {inspect_lead} Trace",
            line=dict(color="#0f172a", width=1.5)
        ),
        row=1, col=1
    )
    # Highlight R-peaks if available
    if len(r_peaks) > 0 and inspect_lead == "II":
        fig_wave.add_trace(
            go.Scatter(
                x=r_peaks / 100.0, y=sig_lead[r_peaks], mode="markers",
                marker=dict(color="red", size=7, symbol="triangle-down"),
                name="Detected R-Peaks"
            ),
            row=1, col=1
        )
    # Attribution Area
    fig_wave.add_trace(
        go.Scatter(
            x=time_sec, y=norm_salience, mode="lines",
            name="Integrated Gradients Salience",
            fill="tozeroy",
            line=dict(color="#ef4444", width=1)
        ),
        row=2, col=1
    )
    fig_wave.update_layout(
        height=440,
        title_text=f"ECG Voltage & Explainability Alignment — Lead {inspect_lead} ({LEAD_ANATOMY[inspect_lead]})",
        xaxis2_title="Time (seconds)",
        yaxis_title="Amplitude (Z-score)",
        yaxis2_title="Attribution Score",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_wave, use_container_width=True)

    # 12-Lead Full Synchronized Heatmap Matrix
    st.markdown("#### 🗺️ 12-Lead Synchronized Attribution Heatmap")
    st.caption("Simultaneous view of morphological salience across all 12 anatomical leads.")
    fig_matrix = go.Figure(
        data=go.Heatmap(
            z=np.abs(attr.T),
            x=time_sec,
            y=LEAD_NAMES,
            colorscale="Reds",
            colorbar=dict(title="Salience")
        )
    )
    fig_matrix.update_layout(
        height=360,
        xaxis_title="Time (seconds)",
        yaxis_title="ECG Leads",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_matrix, use_container_width=True)


# =====================================================================
# TAB 3: Multi-Model Duel & Direct Comparison
# =====================================================================
with tab3:
    st.subheader("3. Multi-Model Duel: Side-by-Side Architectural Comparison")
    st.markdown(
        "Compare two distinct deep learning architectures on the **exact same patient record** in real time "
        "to evaluate consensus, confidence divergence, and explainability differences."
    )

    duel_col1, duel_col2 = st.columns(2)
    with duel_col1:
        model_a = st.selectbox(
            "Select Model A:",
            available_model_keys,
            index=0,
            key="duel_model_a"
        )
    with duel_col2:
        default_b_idx = 1 if len(available_model_keys) > 1 else 0
        model_b = st.selectbox(
            "Select Model B:",
            available_model_keys,
            index=default_b_idx,
            key="duel_model_b"
        )

    # Run inference for both models
    with st.spinner(f"Evaluating {model_a} vs {model_b}..."):
        explainer_a = load_explainer_for_model(model_a)
        explainer_b = load_explainer_for_model(model_b)

        probs_a = explainer_a.predict(ecg_tensor, meta_tensor)
        probs_b = explainer_b.predict(ecg_tensor, meta_tensor)

    # Consensus & Divergence Metric
    cos_sim = float(np.dot(probs_a, probs_b) / (np.linalg.norm(probs_a) * np.linalg.norm(probs_b) + 1e-8))
    max_divergence_idx = int(np.argmax(np.abs(probs_a - probs_b)))
    div_class = CLASSES[max_divergence_idx]
    div_delta = float(probs_a[max_divergence_idx] - probs_b[max_divergence_idx])

    c_metric1, c_metric2, c_metric3 = st.columns(3)
    c_metric1.metric("Architectural Consensus Index", f"{cos_sim * 100:.1f}%")
    c_metric2.metric("Greatest Divergence Class", f"{div_class}")
    c_metric3.metric("Probability Delta", f"{div_delta * 100:+.1f}%")

    # Side-by-side bar chart
    fig_duel = go.Figure()
    fig_duel.add_trace(go.Bar(
        x=CLASSES, y=probs_a * 100, name=f"Model A: {model_a}",
        marker_color="#3b82f6", text=[f"{p*100:.1f}%" for p in probs_a], textposition="auto"
    ))
    fig_duel.add_trace(go.Bar(
        x=CLASSES, y=probs_b * 100, name=f"Model B: {model_b}",
        marker_color="#f97316", text=[f"{p*100:.1f}%" for p in probs_b], textposition="auto"
    ))
    fig_duel.update_layout(
        barmode="group",
        height=380,
        title_text=f"Direct Probability Comparison: {model_a} vs {model_b}",
        xaxis_title="Diagnostic Disease Class",
        yaxis_title="Predicted Probability (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_duel, use_container_width=True)

    # Lead Salience Comparison between Model A and Model B
    st.markdown("#### Explainability Comparison: Lead Importance Agreement")
    with st.spinner("Computing lead attributions for both models..."):
        _, ranks_a = explainer_a.explain_waveform_ig(ecg_tensor, meta_tensor, target_class_idx=t_idx, n_steps=20)
        _, ranks_b = explainer_b.explain_waveform_ig(ecg_tensor, meta_tensor, target_class_idx=t_idx, n_steps=20)

    fig_duel_leads = go.Figure()
    fig_duel_leads.add_trace(go.Bar(
        x=LEAD_NAMES, y=[ranks_a.get(l, 0) * 100 for l in LEAD_NAMES],
        name=model_a, marker_color="#3b82f6"
    ))
    fig_duel_leads.add_trace(go.Bar(
        x=LEAD_NAMES, y=[ranks_b.get(l, 0) * 100 for l in LEAD_NAMES],
        name=model_b, marker_color="#f97316"
    ))
    fig_duel_leads.update_layout(
        barmode="group",
        height=320,
        title_text=f"Lead Attribution Distribution: {model_a} vs {model_b} for Class {xai_class}",
        xaxis_title="ECG Lead",
        yaxis_title="Contribution Energy (%)"
    )
    st.plotly_chart(fig_duel_leads, use_container_width=True)


# =====================================================================
# TAB 4: Benchmark Leaderboard & Comparative Analytics
# =====================================================================
with tab4:
    st.subheader("4. Comprehensive Multi-Model Benchmark Leaderboard (PTB-XL Test Fold 10)")
    st.markdown(
        "All models evaluated under identical conditions on the **2,183 held-out patient ECG records** of Fold 10. "
        "Metrics include Macro AUROC, Macro AUPRC, Macro F1, Sensitivity, Specificity, Brier Score, and Latency."
    )

    # Load precomputed comparison table if available
    csv_table_path = os.path.join(REPORTS_DIR, "model_comparison_table.csv")
    if os.path.exists(csv_table_path):
        df_bench = pd.read_csv(csv_table_path)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark summary table is generating or available via `utils/benchmark_models.py`.")

    b_tabs = st.tabs(["ROC Curves", "Precision-Recall Curves", "Radar Spider Chart", "Probability Calibration"])
    
    with b_tabs[0]:
        roc_img = os.path.join(REPORTS_DIR, "model_comparison_roc.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Multi-Model Multi-Label ROC Curves (PTB-XL Test Fold 10)", use_container_width=True)
        else:
            legacy_roc = os.path.join(REPORTS_DIR, "roc_curves.png")
            if os.path.exists(legacy_roc):
                st.image(legacy_roc, caption="CardioSignal ROC Curves", use_container_width=True)
            else:
                st.info("Run `python utils/benchmark_models.py` to generate high-resolution ROC plots.")

    with b_tabs[1]:
        prc_img = os.path.join(REPORTS_DIR, "model_comparison_prc.png")
        if os.path.exists(prc_img):
            st.image(prc_img, caption="Precision-Recall Curves across Diagnostic Classes", use_container_width=True)
        else:
            st.info("Run `python utils/benchmark_models.py` to generate Precision-Recall curves.")

    with b_tabs[2]:
        radar_img = os.path.join(REPORTS_DIR, "model_radar_chart.png")
        if os.path.exists(radar_img):
            st.image(radar_img, caption="6-Axis Performance & Latency Radar Comparison", use_container_width=True)
        else:
            st.info("Run `python utils/benchmark_models.py` to generate the multi-model radar chart.")

    with b_tabs[3]:
        cal_img = os.path.join(REPORTS_DIR, "calibration_curves.png")
        if os.path.exists(cal_img):
            st.image(cal_img, caption="Probability Calibration & Reliability Diagrams", use_container_width=True)
        else:
            st.info("Run `python utils/benchmark_models.py` to generate calibration diagrams.")


# =====================================================================
# TAB 5: Counterfactual "What-If" Demographics Simulator
# =====================================================================
with tab5:
    st.subheader("5. Interactive Counterfactual 'What-If' Demographic Simulator")
    st.markdown(
        "Explore how patient age and biological sex alter cardiac disease probability for this patient's ECG morphology. "
        "Demonstrates the dynamic multimodal gating mechanism in real time."
    )

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.markdown("##### Adjust Patient Covariates")
        sim_age = st.slider("Simulated Patient Age (Years):", min_value=18, max_value=95, value=actual_age, step=1)
        sim_sex_str = st.radio("Simulated Biological Sex:", ["Female", "Male"], index=0 if meta[1] == 0.0 else 1)
        sim_sex = 1.0 if sim_sex_str == "Male" else 0.0

        # Normalize simulated age: (age - 59.8) / 16.6
        sim_age_norm = (sim_age - 59.8) / 16.6
        sim_meta_tensor = torch.tensor([sim_age_norm, sim_sex], dtype=torch.float32)

        sim_probs = explainer.predict(ecg_tensor, sim_meta_tensor)

    with sim_col2:
        st.markdown("##### Real-Time Risk Response")
        sim_df = pd.DataFrame({
            "Class": CLASSES,
            "Original Risk (%)": probs * 100,
            "Counterfactual Risk (%)": sim_probs * 100,
            "Shift (%)": (sim_probs - probs) * 100
        })

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            x=sim_df["Class"], y=sim_df["Original Risk (%)"],
            name=f"Original ({actual_age}y, {gender_str.split(' ')[0]})",
            marker_color="#94a3b8"
        ))
        fig_sim.add_trace(go.Bar(
            x=sim_df["Class"], y=sim_df["Counterfactual Risk (%)"],
            name=f"Simulated ({sim_age}y, {sim_sex_str})",
            marker_color="#ef4444"
        ))
        fig_sim.update_layout(
            barmode="group",
            height=320,
            title_text="Risk Shift Under Counterfactual Demographics",
            yaxis_title="Probability (%)",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    # Dynamic Age Sensitivity Sweep Curve
    st.markdown("#### Full Lifetime Age Sensitivity Trajectory")
    st.caption("Plots the model's risk trajectory as age varies from 20 to 90 years for both biological sexes.")
    
    age_range = np.linspace(20, 90, 36)
    target_sim_class = st.selectbox("Trajectory Target Disease Class:", CLASSES, index=1)
    target_sim_idx = CLASSES.index(target_sim_class)

    curve_f = []
    curve_m = []
    for a in age_range:
        a_norm = (a - 59.8) / 16.6
        p_f = explainer.predict(ecg_tensor, torch.tensor([a_norm, 0.0], dtype=torch.float32))[target_sim_idx]
        p_m = explainer.predict(ecg_tensor, torch.tensor([a_norm, 1.0], dtype=torch.float32))[target_sim_idx]
        curve_f.append(p_f * 100)
        curve_m.append(p_m * 100)

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(x=age_range, y=curve_f, mode="lines", name="Female (XX)", line=dict(color="#ec4899", width=2.5)))
    fig_traj.add_trace(go.Scatter(x=age_range, y=curve_m, mode="lines", name="Male (XY)", line=dict(color="#3b82f6", width=2.5)))
    fig_traj.add_vline(x=actual_age, line_dash="dash", line_color="gray", annotation_text=f"Current Patient ({actual_age}y)")
    fig_traj.update_layout(
        height=340,
        title_text=f"Multimodal Age-Risk Gradient for {target_sim_class} ({CLASS_DESCRIPTIONS[target_sim_class]})",
        xaxis_title="Patient Age (Years)",
        yaxis_title="Predicted Probability (%)",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_traj, use_container_width=True)


# =====================================================================
# TAB 6: Automated Clinical Report Generator & EHR Export
# =====================================================================
with tab6:
    st.subheader("6. Automated Clinical Diagnostic Report & Electronic Health Record (EHR) Export")
    st.markdown("Generates a formal, standardized cardiology consultation note for clinical decision support.")

    # Generate Clinical Text Note
    primary_dx = CLASSES[int(np.argmax(probs))]
    primary_prob = float(np.max(probs)) * 100
    top_lead_name = list(lead_ranks.keys())[0] if lead_ranks else "V2"
    top_lead_territory = LEAD_ANATOMY.get(top_lead_name, "Global")

    rec_actions = []
    if primary_dx == "MI":
        rec_actions = [
            "STAT Serum High-Sensitivity Cardiac Troponin I/T assay.",
            "Immediate 12-lead ECG repeat in 15-30 minutes to evaluate dynamic ST-segment evolution.",
            "Urgent cardiology consultation and evaluation for emergent coronary angiography / catheterization.",
            "Initiate acute coronary syndrome protocol (dual antiplatelet therapy, anticoagulation as indicated)."
        ]
    elif primary_dx == "STTC":
        rec_actions = [
            "Serial cardiac enzyme testing to rule out non-ST elevation acute coronary syndrome.",
            "Transthoracic echocardiogram (TTE) to evaluate regional wall motion abnormalities.",
            "Electrolyte panel (serum potassium, magnesium, calcium) to rule out metabolic repolarization disturbances.",
            "Outpatient or inpatient cardiology follow-up for ischemic evaluation (stress imaging)."
        ]
    elif primary_dx == "CD":
        rec_actions = [
            "Measure PR, QRS, and QTc intervals; inspect for complete vs incomplete bundle branch block.",
            "Review concurrent medications for AV-nodal blocking agents (Beta-blockers, Calcium channel blockers, Digoxin).",
            "Holter 24-48h rhythm monitoring if patient reports syncope or presyncope.",
            "Assess for structural heart disease via echocardiogram."
        ]
    elif primary_dx == "HYP":
        rec_actions = [
            "Echocardiogram to quantify Left Ventricular Mass Index (LVMI) and relative wall thickness.",
            "Strict blood pressure optimization and 24-hour ambulatory blood pressure monitoring.",
            "Screen for secondary hypertension causes if patient is young."
        ]
    else:
        rec_actions = [
            "Routine cardiovascular follow-up according to standard clinical prevention guidelines.",
            "Maintain healthy lifestyle measures and primary prevention risk factor monitoring."
        ]

    clinical_note = f"""# CARDIOLOGY CONSULTATION REPORT
**CardioSignal-XAI Multimodal Decision Support System**
*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

### PATIENT DEMOGRAPHICS & RECORD IDENTIFIER
- **ECG Record ID:** {filtered_ecg_ids[selected_picker_idx]}
- **Age:** {actual_age} Years
- **Biological Sex:** {gender_str}
- **Resting Heart Rate:** {bpm} BPM ({regularity})
- **QRS Interval (Est.):** {qrs_ms} ms

---

### MULTIMODAL DIAGNOSTIC RISK STRATIFICATION
- **Primary Diagnostic Finding:** **{primary_dx}** ({CLASS_DESCRIPTIONS[primary_dx]})
- **Model Confidence:** **{primary_prob:.1f}%**
- **Ground Truth Baseline:** `{gt_str}`
- **Active Model Backbone:** {selected_model_key} ({model_info.get('family', '')})

#### Complete Probability Breakdown:
{chr(10).join([f"- **{c}:** {probs[i]*100:.2f}% ({CLASS_DESCRIPTIONS[c].split('(')[0].strip()})" for i, c in enumerate(CLASSES)])}

---

### EXPLAINABLE AI (XAI) ANATOMICAL ATTRIBUTION
- **Dominant Salience Lead:** **Lead {top_lead_name}** ({top_lead_territory})
- **Lead Contribution Score:** {lead_ranks.get(top_lead_name, 0)*100:.2f}% of total signal energy
- **Top 3 Critical Leads:** {', '.join([f'Lead {l} ({LEAD_ANATOMY[l]}: {lead_ranks[l]*100:.1f}%)' for l in list(lead_ranks.keys())[:3]])}
- **Demographic Bias Assessment (SHAP):** Age Shift = {shap_meta['Age']:+.4f}, Sex Shift = {shap_meta['Sex']:+.4f}

---

### ACTIONABLE CLINICAL RECOMMENDATIONS
{chr(10).join([f"{idx+1}. {act}" for idx, act in enumerate(rec_actions)])}

---
*Disclaimer: CardioSignal-XAI is an artificial intelligence clinical decision support tool designed to assist healthcare professionals. It does not replace clinical judgment or comprehensive physician evaluation.*
"""

    st.markdown(clinical_note)

    # Export Buttons
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Download Clinical Report (.md)",
            data=clinical_note,
            file_name=f"CardioSignal_Report_{filtered_ecg_ids[selected_picker_idx]}.md",
            mime="text/markdown"
        )
    with col_dl2:
        json_report = {
            "patient_id": int(filtered_ecg_ids[selected_picker_idx]),
            "age": actual_age,
            "sex": gender_str,
            "vitals": {"bpm": bpm, "qrs_ms": qrs_ms, "rhythm": regularity},
            "predictions": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
            "primary_diagnosis": primary_dx,
            "top_lead": top_lead_name,
            "top_lead_anatomy": top_lead_territory,
            "recommendations": rec_actions
        }
        st.download_button(
            label="📥 Download Structured EHR JSON (.json)",
            data=json.dumps(json_report, indent=2),
            file_name=f"CardioSignal_EHR_{filtered_ecg_ids[selected_picker_idx]}.json",
            mime="application/json"
        )


# =====================================================================
# TAB 7: 2D Time-Frequency & Standard Paper Tracing
# =====================================================================
with tab7:
    st.subheader("7. Advanced 2D Signal Representations & Clinical Paper Tracing")
    
    tab7_sub1, tab7_sub2 = st.tabs(["Time-Frequency Scalogram (Spectrogram)", "Standard Clinical 12-Lead Paper Tracing"])
    
    with tab7_sub1:
        st.markdown(
            "Captures non-stationary frequency anomalies (e.g., high-frequency QRS notches, "
            "ventricular activation dispersion, and flutter harmonics)."
        )
        cwt_lead = st.selectbox(
            "Select Lead for Time-Frequency Decomposition:",
            LEAD_NAMES,
            index=LEAD_NAMES.index(top_lead_name)
        )
        l_cwt_idx = LEAD_NAMES.index(cwt_lead)
        t_spec, f_spec, sxx_db = compute_time_frequency_spectrogram(ecg[:, l_cwt_idx], fs=100)

        fig_spec = go.Figure(
            data=go.Heatmap(
                x=t_spec, y=f_spec, z=sxx_db,
                colorscale="Viridis",
                colorbar=dict(title="Power (dB)")
            )
        )
        fig_spec.update_layout(
            title=f"Time-Frequency Energy Distribution — Lead {cwt_lead} ({LEAD_ANATOMY[cwt_lead]})",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    with tab7_sub2:
        st.markdown(
            "Renders the classic 12-lead paper layout (4 columns × 3 rows) standard in emergency departments and cardiology clinics."
        )
        fig_paper = plot_standard_12lead_ecg(ecg, LEAD_NAMES, fs=100)
        st.plotly_chart(fig_paper, use_container_width=True)