# CardioSignal-XAI 🫀

> **A Multimodal Explainable Deep Learning Framework for Cardiac Disease Risk Stratification Using 12-Lead ECG Signals and Patient Metadata**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 🌟 Key Novelties
1. **Cross-Attention Multimodal Fusion:** Fuses raw 12-lead time-series morphology (1D-CNN + BiLSTM) with demographic risk embeddings (Age, Sex), dynamically weighting waveform features.
2. **Dual-Level Explainability (XAI):** 
   - Waveform-level attributions via **Integrated Gradients** (Captum).
   - Demographic risk shift analysis via **KernelSHAP**.
3. **Anatomical Lead-Wise Spatial Ranking:** Quantifies spatial energy (L2-norm) across individual leads ($V_1–V_6$, $I, II, III$, etc.) to align AI output directly with clinical cardiology workflows.

---

## 📊 Experimental Results (PTB-XL Test Fold 10)
- **Macro AUROC:** **0.9053**
- **Macro F1:** **0.6678**
- **Per-Class AUROC:** NORM: `0.9446` | MI: `0.9162` | STTC: `0.9310` | CD: `0.9172` | HYP: `0.8177`

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/CardioSignal-XAI.git
cd CardioSignal-XAI
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt