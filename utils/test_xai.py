import h5py
import torch
import numpy as np
from xai.explainer import CardioExplainer, CLASSES

H5_PATH = "data/processed/ptbxl_preprocessed.h5"

def test_xai():
    print("[*] Initializing CardioExplainer...")
    explainer = CardioExplainer()

    with h5py.File(H5_PATH, "r") as f:
        # Load sample from test fold (Fold 10)
        folds = f["folds"][:]
        test_indices = np.where(folds == 10)[0]
        sample_idx = test_indices[0]

        ecg = f["ecg_signals"][sample_idx].T  # (12, 1000)
        meta = f["metadata"][sample_idx]      # (2,)
        label = f["labels"][sample_idx]        # (5,)
        background_meta = f["metadata"][:50]

    ecg_t = torch.tensor(ecg, dtype=torch.float32)
    meta_t = torch.tensor(meta, dtype=torch.float32)

    # 1. Test Prediction
    probs = explainer.predict(ecg_t, meta_t)
    print("\n[✓] Diagnostic Predictions:")
    for c, p, gt in zip(CLASSES, probs, label):
        print(f"    - {c:5s}: {p*100:5.2f}% (Ground Truth: {int(gt)})")

    # 2. Test Waveform Integrated Gradients & Lead-Wise Importance
    target_class = int(np.argmax(probs))
    print(f"\n[*] Generating Waveform Attributions for: {CLASSES[target_class]}...")
    attr, lead_ranks = explainer.explain_waveform_ig(ecg_t, meta_t, target_class_idx=target_class)
    print(f"[✓] Attribution Matrix Shape: {attr.shape}")
    print("[✓] Top 3 Contributing Leads:")
    for lead, score in list(lead_ranks.items())[:3]:
        print(f"    - Lead {lead:4s}: {score*100:5.2f}% energy")

    # 3. Test Demographic SHAP
    print("\n[*] Generating Demographic SHAP Attributions...")
    shap_meta = explainer.explain_demographics_shap(background_meta, meta, target_class_idx=target_class)
    print(f"[✓] SHAP Values: {shap_meta}")

if __name__ == "__main__":
    test_xai()