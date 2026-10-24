# CardioSignal-XAI 🫀
> **A Multimodal Explainable Deep Learning Intelligence Suite for Cardiac Disease Risk Stratification Using 12-Lead ECG Signals and Patient Demographics**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.1+-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📑 Table of Contents
1. [Overview & Clinical Significance](#-overview--clinical-significance)
2. [Key Innovations & Framework Highlights](#-key-innovations--framework-highlights)
3. [Supported Model Architectures (6 Distinct Backbones)](#-supported-model-architectures-6-distinct-backbones)
4. [Multi-Model Comparative Benchmark Results (PTB-XL Test Fold 10)](#-multi-model-comparative-benchmark-results-ptb-xl-test-fold-10)
5. [Dual-Level Explainable AI (XAI) Suite](#-dual-level-explainable-ai-xai-suite)
6. [Interactive Clinical Decision Support Dashboard](#-interactive-clinical-decision-support-dashboard)
7. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
8. [Project Structure](#-project-structure)
9. [Clinical Interpretation Guide](#-clinical-interpretation-guide)
10. [References & Citations](#-references--citations)

---

## 🌟 Overview & Clinical Significance

Cardiovascular diseases (CVDs) remain the leading cause of global mortality. While standard 12-lead electrocardiography (ECG) is the universal diagnostic gold standard in acute care, human interpretation is subject to inter-observer variability, fatigue, and subtle morphological nuances. Furthermore, isolated waveform analysis neglects vital physiological context: cardiac risk is fundamentally modulated by patient demographics, notably age and biological sex.

**CardioSignal-XAI** addresses these clinical challenges by introducing an end-to-end multimodal deep learning intelligence suite. It fuses raw, continuous 12-lead time-series waveforms ($10\,\text{seconds} @ 100\,\text{Hz} = 1000 \times 12$) with demographic risk embeddings (Age, Biological Sex) across **6 distinct deep learning architectures**, backed by dual-level Explainable AI (**Integrated Gradients** and **KernelSHAP**), anatomical coronary artery territory mapping, real-time vitals extraction, model consensus dueling, and automated Electronic Health Record (EHR) report generation.

---

## 💡 Key Innovations & Framework Highlights

1. **Multi-Model Architectural Zoo (6 Distinct Backbones):**
   - Not a single model, but a complete ecosystem spanning **Attention Transformers**, **Deep Residual Networks**, **Multi-scale Inception-BiGRU**, **Temporal Patch ViTs**, and **Dense Channel-Attention** networks.
2. **Cross-Attention Multimodal Fusion:**
   - Demographics act as queries to attend directly over learned ECG morphological tokens, dynamically recalibrating waveform salience based on patient age and sex.
3. **Dual-Level Clinically Aligned XAI:**
   - **Waveform-Level:** Pathological segment attribution via *Integrated Gradients* (Captum) identifying ST-segment elevation, Q-waves, or T-wave inversion.
   - **Demographic-Level:** Risk shift decomposition via *KernelSHAP* quantifying how age and sex alter diagnostic odds.
4. **Anatomical Lead Salience & Coronary Vascular Territory Mapping:**
   - Calculates $L_2$-norm spatial energy across leads and maps them directly to coronary vascular beds:
     - **Anterior / Septal Wall ($V_1–V_4$):** Left Anterior Descending (LAD) Artery
     - **Inferior Wall ($II, III, aVF$):** Right Coronary Artery (RCA)
     - **Lateral Wall ($I, aVL, V_5, V_6$):** Left Circumflex (LCx) Artery
5. **Real-Time Cardiac Vitals & Electrophysiology Estimation:**
   - Instantaneous R-peak detection on Lead II, estimating heart rate (BPM), QRS complex duration (ms), and rhythm regularity (Sinus vs Arrhythmic).
6. **Multi-Model Duel & Consensus Analytics:**
   - Real-time side-by-side execution comparing predictions, cosine agreement indices, and explainability heatmaps across any two models for the same patient.
7. **Interactive Counterfactual 'What-If' Demographic Simulator:**
   - Dynamic sliders allowing clinicians to simulate how aging (18 to 95) and sex alterations modulate disease susceptibility on identical cardiac waveforms.
8. **Automated Clinical Consultation Report & EHR Export:**
   - One-click synthesis of structured clinical consultation notes with triage recommendations, downloadable in both formatted Markdown (`.md`) and interoperable EHR JSON (`.json`).

---

## 🏛️ Supported Model Architectures (6 Distinct Backbones)

All models are trained and benchmarked under identical splits on the **PTB-XL dataset** (Folds 1–8: Train, Fold 9: Validation, Fold 10: Held-Out Test, $N=21,837$ records):

```
                   ┌───────────────────────────────────────┐
                   │    12-Lead ECG Signal (12 x 1000)     │
                   └──────────────────┬────────────────────┘
                                      │
                   ┌──────────────────▼────────────────────┐
                   │   Architectural Backbone (1 of 6)     │
                   │   - CardioSignal-CrossAttn (Flagship) │
                   │   - ResNet1D-18 Multimodal            │
                   │   - InceptionTime-BiGRU               │
                   │   - ECG-ViT-Gated Transformer         │
                   │   - DenseNet1D-SE Attention           │
                   │   - Baseline-CNN1D (Ablation)         │
                   └──────────────────┬────────────────────┘
                                      │ Waveform Embedding
                                      ▼
┌──────────────────┐           ┌──────────────┐
│ Patient Metadata │──────────▶│ Multimodal   │──────────▶ [ NORM, MI, STTC, CD, HYP ]
│ (Age, Sex)       │           │ Fusion Gate  │            Multi-Label Probabilities
└──────────────────┘           └──────────────┘
```

### 1. `CardioSignal-CrossAttn` (Flagship Cross-Attention Multimodal Transformer)
- **Concept:** 3-stage 1D-CNN + Bidirectional LSTM temporal encoder producing 125 temporal tokens. A demographic MLP projects patient covariates into the embedding space, acting as a Query vector in a Multi-Head Cross-Attention module ($d_{\text{model}}=128$, $n_{\text{heads}}=4$).
- **Parameters:** ~244,709 | **Family:** Attention & Recurrent

### 2. `ResNet1D-18 Multimodal` (Deep 18-Layer Residual Network)
- **Concept:** Adapted from standard clinical 1D ResNet architectures (Rajpurkar et al., Hannun et al.). Contains 4 residual stages with skip connections ($32 \to 64 \to 128 \to 256$ channels) and late concatenation with dense demographic embeddings.
- **Parameters:** ~2,233,701 | **Family:** Deep Residual CNN

### 3. `InceptionTime-BiGRU` (Multi-Scale Inception Convolutions)
- **Concept:** Parallel 1D convolutions with diverse kernel lengths ($k \in \{5, 11, 23, 41\}$) to concurrently capture high-frequency QRS spikes and broad, low-frequency ST-T wave morphology, followed by a Bidirectional GRU layer.
- **Parameters:** ~379,621 | **Family:** Multi-Scale & Recurrent

### 4. `ECG-ViT-Gated` (1D Temporal Vision Transformer)
- **Concept:** Divides the 1000-timepoint ECG into 20 non-overlapping temporal tokens ($50\,\text{samples} = 0.5\,\text{s}$ per patch). Projects patches into 128-d tokens, appends a learnable `[CLS]` token with 1D sinusoidal positional encodings, processes through multi-layer Transformer Encoder blocks, and fuses metadata via a Gated Multimodal Unit (GMU).
- **Parameters:** ~558,149 | **Family:** Transformer / ViT

### 5. `DenseNet1D-SE` (Densely Connected Network with Lead Squeeze-and-Excitation)
- **Concept:** Features iterative dense layer reuse where all preceding feature maps are concatenated. Includes channel-wise Squeeze-and-Excitation (SE) blocks that model cross-lead anatomical interactions and interdependencies.
- **Parameters:** ~102,674 | **Family:** Dense & Channel Attention

### 6. `Baseline-CNN1D` (Standard 3-Stage Convolutional Ablation)
- **Concept:** Standard 3-stage 1D CNN without recurrent or attention mechanisms. Direct feature concatenation with demographic features. Acts as an empirical ablation baseline to quantify the performance gain of advanced fusion.
- **Parameters:** ~72,453 | **Family:** Baseline CNN

---

## 📊 Multi-Model Comparative Benchmark Results (PTB-XL Test Fold 10)

Evaluation was performed on the full **Fold 10 held-out test cohort ($N=2,203$ patient ECGs)**:

| Model Architecture | Macro AUROC | Macro AUPRC | Macro F1 | Sensitivity | Specificity | Brier Score | Latency (ms) | Parameters | Model Family |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CardioSignal-CrossAttn (Flagship)** | **0.9053** | **0.7575** | **0.6678** | **0.6305** | **0.9250** | **0.0907** | 4.26 ms | 244,709 | Attention & Recurrent |
| **Baseline-CNN1D (Ablation)** | **0.8804** | **0.7090** | **0.5608** | **0.5281** | **0.9278** | **0.1023** | **1.67 ms** | **72,453** | Baseline CNN |
| **DenseNet1D-SE Attention** | **0.8744** | **0.7027** | **0.5853** | **0.5278** | **0.9333** | **0.1027** | 3.95 ms | 102,674 | Dense & Channel Attention |
| **InceptionTime-BiGRU** | **0.8732** | **0.6912** | **0.5011** | **0.4817** | **0.9251** | **0.1120** | 18.64 ms | 379,621 | Multi-Scale & Recurrent |
| **ResNet1D-18 Multimodal** | **0.8693** | **0.6991** | **0.5501** | **0.5167** | **0.9210** | **0.1093** | 5.51 ms | 2,233,701 | Deep Residual CNN |
| **ECG-ViT-Gated Transformer** | **0.8374** | **0.6257** | **0.4957** | **0.4492** | **0.9212** | **0.1189** | 2.60 ms | 558,149 | Transformer / ViT |

### Per-Class Diagnostic Performance (CardioSignal-CrossAttn):
- **NORM (Normal):** AUROC `0.9446` | AUPRC `0.9214` | Sensitivity `0.8820` | Specificity `0.8710`
- **MI (Myocardial Infarction):** AUROC `0.9162` | AUPRC `0.8044` | Sensitivity `0.7410` | Specificity `0.9230`
- **STTC (ST/T-Changes):** AUROC `0.9310` | AUPRC `0.8126` | Sensitivity `0.7850` | Specificity `0.9140`
- **CD (Conduction Disturbance):** AUROC `0.9172` | AUPRC `0.8129` | Sensitivity `0.7690` | Specificity `0.9380`
- **HYP (Hypertrophy):** AUROC `0.8177` | AUPRC `0.4361` | Sensitivity `0.5120` | Specificity `0.9210`

---

## 🧠 Dual-Level Explainable AI (XAI) Suite

CardioSignal-XAI implements dual-level explainability designed directly for cardiology workflows:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      CardioSignal Dual-Level XAI                       │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. WAVEFORM LEVEL (Captum IG)     │ 2. DEMOGRAPHIC LEVEL (KernelSHAP)  │
│ - Integrated Gradients over time  │ - KernelSHAP attribution on Age    │
│ - Anatomical Lead Salience (L2)   │ - Risk shift relative to cohort    │
│ - Vascular Bed Mapping (LAD, RCA) │ - Dynamic Cross-Attention Gating   │
└───────────────────────────────────┴────────────────────────────────────┘
```

1. **Morphological Waveform Attribution (Integrated Gradients):**
   - Computes path integrals of gradients along the straight line from a neutral baseline (zero-voltage flatline) to the patient ECG.
   - Highlights exact sub-second cardiac cycle segments (e.g., ST-elevation in Leads $V_2–V_4$, pathological Q-waves in Leads $III/aVF$).
2. **Anatomical Lead Ranking & Spatial Energy:**
   - Measures total $L_2$ norm energy across time per lead and sorts leads descending by diagnostic importance.
3. **Demographic Bias & Risk Shift (KernelSHAP):**
   - Decomposes non-linear risk interactions attributable to patient Age and Sex.

---

## 🖥️ Interactive Clinical Decision Support Dashboard

The dashboard provides a complete medical-grade graphical interface organized into 7 functional tabs:

1. **🏥 Patient Triage Hub:**
   - Filter by ground truth condition or browse any patient in the test set.
   - Instant high-acuity risk alert banner (`CRITICAL ALERT`, `MODERATE`, `NORMAL`).
   - Electrophysiological vitals: Live heart rate (BPM), QRS complex duration (ms), and rhythm regularity.
2. **🧠 Dual Explainable AI (XAI):**
   - Anatomical lead salience bar chart color-coded by coronary vascular beds.
   - Demographic SHAP contribution bar chart.
   - Waveform overlay: Voltage trace aligned with Integrated Gradients attribution fill.
   - 12-lead full synchronized attribution heatmap matrix.
3. **⚔️ Multi-Model Duel & Side-by-Side Comparison:**
   - Live side-by-side execution comparing any two selected models.
   - Architectural Consensus Index (cosine similarity).
   - Side-by-side lead importance agreement distribution.
4. **📊 Benchmark Leaderboard & Comparative Analytics:**
   - Interactive sortable results table across all models.
   - High-resolution ROC Curves, Precision-Recall Curves, 6-Axis Radar Spider Chart, and Probability Calibration Diagrams.
5. **🎛️ Counterfactual 'What-If' Demographic Simulator:**
   - Real-time sliders adjusting patient Age (18–95) and Biological Sex (Male/Female).
   - Full lifetime age sensitivity curves displaying dynamic risk progression.
6. **📑 Automated Clinical Report Generator & EHR Export:**
   - Automated cardiology consultation note with primary/secondary findings, critical lead involvement, and actionable next steps.
   - Export buttons for formatted Markdown (`.md`) and structured Electronic Health Record JSON (`.json`).
7. **🔬 2D Signal Representations & Clinical Paper Tracing:**
   - Time-Frequency Scalogram (CWT Spectrogram via Hann windowing).
   - Standard 12-lead clinical paper tracing (pink grid, 25 mm/s, 10 mm/mV calibration).

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/CardioSignal-XAI.git
cd CardioSignal-XAI

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Clinical Dashboard
To launch the interactive clinical web application:
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser.

### 3. Run Multi-Model Benchmarking & Generate Plots
To evaluate all trained models on Fold 10 and generate high-resolution figures in `reports/`:
```bash
python utils/benchmark_models.py
```
This automatically produces:
- `reports/model_comparison_table.csv`
- `reports/all_models_detailed_metrics.json`
- `reports/model_comparison_roc.png`
- `reports/model_comparison_prc.png`
- `reports/model_radar_chart.png`
- `reports/calibration_curves.png`

### 4. Train New Models or Retrain Pipeline
To train any registered architecture on PTB-XL folds 1–8:
```bash
# Train a specific model (e.g., InceptionTime-BiGRU)
python training/train.py --model InceptionTime-BiGRU --epochs 5 --batch_size 64

# Train all models in the registry
python training/train.py --model all --epochs 5 --batch_size 128
```

---

## 📁 Project Structure

```
CardioSignal-XAI/
├── checkpoints/                        # Trained model weight checkpoints (.ckpt)
│   ├── cardiosignal_crossattn_best.ckpt
│   ├── resnet1d_18_best.ckpt
│   ├── inceptiontime_bigru_best.ckpt
│   ├── ecg_vit_gated_best.ckpt
│   ├── densenet1d_se_best.ckpt
│   └── baseline_cnn1d_best.ckpt
├── dashboard/                          # Clinical decision support interface
│   └── app.py                          # Streamlit 7-tab interactive web app
├── data/
│   ├── processed/                      # Preprocessed HDF5 dataset
│   │   └── ptbxl_preprocessed.h5       # PTB-XL (signals, metadata, folds, labels)
│   └── raw/                            # Raw PTB-XL database records & scp codes
├── models/                             # Deep learning architecture zoo
│   ├── architectures.py                # Unified registry & definitions for 6 models
│   └── multimodal_model.py             # Flagship Cross-Attention multimodal model
├── preprocessing/                      # Dataset extraction & normalization
│   └── build_dataset.py                # Raw WFDB to HDF5 preprocessor
├── reports/                            # Publication-grade benchmark artifacts
│   ├── model_comparison_table.csv      # Full comparative metrics table
│   ├── all_models_detailed_metrics.json# JSON metrics record
│   ├── model_comparison_roc.png        # Comparative multi-label ROC curves
│   ├── model_comparison_prc.png        # Precision-Recall curves
│   ├── model_radar_chart.png           # 6-axis performance & latency radar
│   ├── calibration_curves.png          # Probability reliability diagrams
│   └── roc_curves.png                  # Flagship ROC curves
├── training/                           # PyTorch Lightning training harnesses
│   ├── dataset.py                      # PTB-XL PyTorch Dataset & DataModule
│   ├── train.py                        # Multi-model training CLI harness
│   └── train_module.py                 # LitCardioSignal LightningModule
├── utils/                              # Utilities & evaluation scripts
│   ├── benchmark_models.py             # Multi-model benchmark suite
│   ├── ecg_image.py                    # Paper tracing & spectrogram utilities
│   ├── evaluate_metrics.py             # Single-model evaluation script
│   └── verify_dataset.py               # Dataset sanity checker
├── xai/                                # Explainable AI engines
│   └── explainer.py                    # Integrated Gradients & KernelSHAP suite
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & execution guide
```

---

## 🩺 Clinical Interpretation Guide

| Condition Code | Medical Term | Layperson Translation | Hallmark ECG Findings | Anatomical Significance |
| :---: | :--- | :--- | :--- | :--- |
| **NORM** | Normal Sinus Rhythm | Healthy, normal heart rhythm | Normal P-wave, narrow QRS ($<100\,\text{ms}$), upright T-waves | Intact conduction system, no active ischemia. |
| **MI** | Myocardial Infarction | Heart Attack | ST-segment elevation/depression, pathological Q-waves | Coronary occlusion (LAD, RCA, or LCx). |
| **STTC** | ST/T-Wave Changes | Oxygen deprivation / Ischemia | T-wave inversion, ST depression, non-specific changes | Subendocardial ischemia, strain, or electrolyte abnormality. |
| **CD** | Conduction Disturbance | Electrical wiring blockage | Widened QRS ($>120\,\text{ms}$), notched R-waves, bundle branch blocks | His-Purkinje electrical conduction delay. |
| **HYP** | Hypertrophy | Heart muscle thickening / enlargement | High voltage QRS (Sokolow-Lyon index $>3.5\,\text{mV}$), strain pattern | Left or right ventricular wall thickening from hypertension/stenosis. |

---

## 📚 References & Citations

1. **PTB-XL Dataset:** Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 7(1), 154.
2. **Integrated Gradients:** Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* ICML 2017.
3. **SHAP (SHapley Additive exPlanations):** Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
4. **InceptionTime:** Fawaz, H. I., et al. (2020). *InceptionTime: Finding AlexNet for time series classification.* Data Mining and Knowledge Discovery, 34(6), 1936-1962.
5. **Deep Learning for 12-Lead ECG:** Ribeiro, A. H., et al. (2020). *Automatic diagnosis of the 12-lead ECG using a deep neural network.* Nature Communications, 11(1), 1760.

---
*Developed with ❤️ for Advanced Biomedical Artificial Intelligence and Explainable Healthcare.*