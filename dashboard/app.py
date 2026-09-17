import os
import sys
import glob
from pathlib import Path

# --- 1. Dynamic Root Path Injection ---
# Ensures relative imports from 'models', 'xai', and 'utils' work across all execution contexts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xai.explainer import CardioExplainer, CLASSES, LEAD_NAMES

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="CardioSignal-XAI Clinical Dashboard",
    page_icon="🫀",
    layout="wide"
)

H5_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ptbxl_preprocessed.h5")


# --- 3. Resource & Data Loaders ---
@st.cache_resource
def get_explainer():
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        st.error(f"No trained checkpoint found in {ckpt_dir}")
        st.stop()
    return CardioExplainer(ckpt_files[0])


@st.cache_data
def load_patient_index():
    if not os.path.exists(H5_PATH):
        st.error(f"Preprocessed dataset not found at: {H5_PATH}")
        st.stop()
    with h5py.File(H5_PATH, "r") as f:
        folds = f["folds"][:]
        test_indices = np.where(folds == 10)[0]
        ecg_ids = f["ecg_ids"][test_indices]
    return test_indices, ecg_ids


def load_patient_data(dataset_idx):
    with h5py.File(H5_PATH, "r") as f:
        ecg = f["ecg_signals"][dataset_idx]        # Shape: (1000, 12)
        meta = f["metadata"][dataset_idx]          # Shape: (2,) -> [Age, Sex]
        labels = f["labels"][dataset_idx]          # Shape: (5,)
        bg_meta = f["metadata"][:50]               # Baseline background cohort for SHAP
    return ecg, meta, labels, bg_meta


# --- 4. Main Application Header ---
st.title("🫀 CardioSignal-XAI: Clinical Decision Support & Explainability")
st.markdown(
    "Multimodal 12-Lead ECG & Patient Demographics Risk Stratification with "
    "**Cross-Attention Fusion**, **Integrated Gradients**, and **Lead-Wise Attribution**."
)

explainer = get_explainer()
test_indices, ecg_ids = load_patient_index()

# --- 5. Sidebar: Patient Selection & Clinical Metadata ---
st.sidebar.header("Patient Record Selection")
selected_id_idx = st.sidebar.selectbox(
    "Select Test Patient (PTB-XL Fold 10)",
    range(len(ecg_ids)),
    format_func=lambda x: f"ECG ID: {ecg_ids[x]}"
)
current_dataset_idx = test_indices[selected_id_idx]
ecg, meta, ground_truth, bg_meta = load_patient_data(current_dataset_idx)

# Demographics display
gender_str = "Male" if meta[1] == 1.0 else "Female"
st.sidebar.markdown(f"**Sex:** {gender_str}")
st.sidebar.markdown(f"**Normalized Age:** {meta[0]:.2f}")

# Ground truth display
gt_conditions = [CLASSES[i] for i, val in enumerate(ground_truth) if val == 1.0]
st.sidebar.markdown(f"**Ground Truth Diagnosis:** {', '.join(gt_conditions) if gt_conditions else 'None'}")

# --- 6. Model Inference ---
ecg_tensor = torch.tensor(ecg.T, dtype=torch.float32)  # Shape: (12, 1000)
meta_tensor = torch.tensor(meta, dtype=torch.float32)  # Shape: (2,)
probs = explainer.predict(ecg_tensor, meta_tensor)

# --- 7. Section 1: Diagnostic Risk Stratification ---
st.subheader("1. Diagnostic Risk Stratification")
cols = st.columns(len(CLASSES))
for i, col in enumerate(cols):
    cls_name = CLASSES[i]
    p_val = probs[i]
    is_alert = p_val >= 0.5
    col.metric(
        label=cls_name,
        value=f"{p_val * 100:.1f}%",
        delta="HIGH RISK" if is_alert else "LOW RISK",
        delta_color="inverse" if is_alert else "normal"
    )

st.divider()

# --- 8. Section 2: Dual Explainable AI (XAI) & Spatial Lead Salience ---
st.subheader("2. Dual Explainable AI (XAI) & Spatial Lead Salience")
xai_target_class = st.selectbox(
    "Explain Target Diagnostic Class:",
    CLASSES,
    index=int(np.argmax(probs))
)
target_idx = CLASSES.index(xai_target_class)

with st.spinner("Computing Integrated Gradients & Demographic SHAP..."):
    attr, lead_ranks = explainer.explain_waveform_ig(ecg_tensor, meta_tensor, target_class_idx=target_idx)
    shap_meta = explainer.explain_demographics_shap(bg_meta, meta, target_class_idx=target_idx)

col_xai_left, col_xai_right = st.columns([2, 1])

# Left Column: Lead-wise Attribution Bar Chart
with col_xai_left:
    st.markdown("#### Anatomical Lead Importance (Spatial Energy Ranking)")
    leads_df = pd.DataFrame(list(lead_ranks.items()), columns=["Lead", "Importance"])
    fig_leads = go.Figure(
        go.Bar(
            x=leads_df["Importance"] * 100,
            y=leads_df["Lead"],
            orientation="h",
            marker=dict(color="crimson")
        )
    )
    fig_leads.update_layout(
        xaxis_title="Contribution Energy (%)",
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_leads, use_container_width=True)

# Right Column: Demographic SHAP Impact
with col_xai_right:
    st.markdown("#### Demographic Contribution (SHAP)")
    shap_df = pd.DataFrame({
        "Feature": ["Age", "Sex"],
        "SHAP Impact": [shap_meta["Age"], shap_meta["Sex"]]
    })
    fig_shap = go.Figure(
        go.Bar(
            x=shap_df["SHAP Impact"],
            y=shap_df["Feature"],
            orientation="h",
            marker=dict(color=["royalblue" if v > 0 else "gray" for v in shap_df["SHAP Impact"]])
        )
    )
    fig_shap.update_layout(
        xaxis_title="Marginal Risk Contribution",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()


# --- 9. Section 3: Isolated Lead Waveform Inspector ---
@st.fragment
def render_waveform_fragment(signal_data, attribution_data, default_lead):
    st.subheader("3. 12-Lead Morphological Waveform Attribution Overlay")
    inspect_lead = st.selectbox(
        "Select Lead to Inspect:",
        LEAD_NAMES,
        index=LEAD_NAMES.index(default_lead)
    )
    lead_i = LEAD_NAMES.index(inspect_lead)

    time_axis = np.linspace(0, 10, 1000)
    sig = signal_data[:, lead_i]
    salience = np.abs(attribution_data[:, lead_i])
    salience_norm = (salience - salience.min()) / (salience.max() - salience.min() + 1e-8)

    fig_ecg = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08
    )

    # Raw ECG Lead Trace
    fig_ecg.add_trace(
        go.Scatter(
            x=time_axis,
            y=sig,
            mode="lines",
            name=f"Lead {inspect_lead}",
            line=dict(color="black", width=1.5)
        ),
        row=1, col=1
    )

    # Integrated Gradients Salience Area Chart
    fig_ecg.add_trace(
        go.Scatter(
            x=time_axis,
            y=salience_norm,
            mode="lines",
            name="Integrated Gradients Salience",
            fill="tozeroy",
            line=dict(color="red", width=1)
        ),
        row=2, col=1
    )

    fig_ecg.update_layout(
        height=450,
        title_text=f"ECG Waveform & Attribution Alignment for Lead {inspect_lead}",
        xaxis2_title="Time (seconds)",
        yaxis_title="Amplitude (Z-score)",
        yaxis2_title="Attribution",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_ecg, use_container_width=True)


top_lead = list(lead_ranks.keys())[0] if lead_ranks else LEAD_NAMES[0]
render_waveform_fragment(ecg, attr, top_lead)