import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import glob
import torch
import numpy as np
import shap
from captum.attr import IntegratedGradients

from models.architectures import MODEL_REGISTRY, build_model
from training.train_module import LitCardioSignal

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_DESCRIPTIONS = {
    "NORM": "Normal Sinus Rhythm (Healthy cardiac electrical conduction)",
    "MI": "Myocardial Infarction (Acute or previous heart attack / necrosis)",
    "STTC": "ST/T Wave Changes (Ischemia, repolarization anomaly, strain)",
    "CD": "Conduction Disturbance (Bundle branch block, AV block, fascicular block)",
    "HYP": "Hypertrophy (Ventricular or atrial wall thickening/enlargement)"
}

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ANATOMY = {
    "I": "Lateral wall",
    "II": "Inferior wall",
    "III": "Inferior wall",
    "aVR": "Cavity / Basal septum",
    "aVL": "High lateral wall",
    "aVF": "Inferior wall",
    "V1": "Septal / Right ventricle",
    "V2": "Septal / Anterior wall",
    "V3": "Anterior wall",
    "V4": "Anterior / Apical wall",
    "V5": "Lateral wall",
    "V6": "Lateral wall"
}


class CardioExplainer:
    """
    Model-agnostic explainability suite supporting all registered multimodal ECG architectures.
    Provides Integrated Gradients (Captum) for 12-lead morphology and KernelSHAP for patient demographics.
    """
    def __init__(self, checkpoint_path=None, model_key="CardioSignal-CrossAttn", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_key = model_key

        if checkpoint_path is None:
            # Try to find model-specific checkpoint first
            safe_slug = model_key.lower().replace("-", "_").replace(" ", "_")
            pattern = os.path.join(PROJECT_ROOT, "checkpoints", f"*{safe_slug}*.ckpt")
            matches = glob.glob(pattern)
            if not matches:
                # Fallback to any checkpoint
                all_ckpts = glob.glob(os.path.join(PROJECT_ROOT, "checkpoints", "*.ckpt"))
                if not all_ckpts:
                    raise FileNotFoundError(f"No checkpoint found in checkpoints/")
                checkpoint_path = all_ckpts[0]
            else:
                checkpoint_path = matches[0]

        self.checkpoint_path = checkpoint_path
        self._load_model()
        self.ig = IntegratedGradients(self._forward_target)

    def _load_model(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        hparams = ckpt.get("hyper_parameters", {})
        detected_model_key = hparams.get("model_key", self.model_key)
        self.model_key = detected_model_key

        try:
            lit_module = LitCardioSignal.load_from_checkpoint(self.checkpoint_path, map_location=self.device)
            self.model = lit_module.model.to(self.device)
        except Exception:
            # Direct weights loading fallback
            self.model = build_model(self.model_key, num_classes=len(CLASSES)).to(self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            clean_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k}
            if clean_dict:
                self.model.load_state_dict(clean_dict, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def _forward_target(self, ecg, metadata):
        return self.model(ecg, metadata)

    def predict(self, ecg_tensor, meta_tensor):
        """
        Computes calibrated disease probabilities across all 5 classes.
        ecg_tensor: (12, 1000) or (1, 12, 1000)
        meta_tensor: (2,) or (1, 2)
        Returns: numpy array of shape (5,)
        """
        if ecg_tensor.dim() == 2:
            ecg_tensor = ecg_tensor.unsqueeze(0)
        if meta_tensor.dim() == 1:
            meta_tensor = meta_tensor.unsqueeze(0)

        with torch.no_grad():
            ecg_in = ecg_tensor.to(self.device)
            meta_in = meta_tensor.to(self.device)
            logits = self.model(ecg_in, meta_in)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probs

    def explain_waveform_ig(self, ecg_tensor, meta_tensor, target_class_idx, n_steps=25):
        """
        Integrated Gradients attribution over 12-lead ECG signals.
        Returns:
            attr_np: (1000, 12) temporal salience array
            lead_importance: dict mapping lead name to normalized contribution energy (%)
        """
        if ecg_tensor.dim() == 2:
            ecg_tensor = ecg_tensor.unsqueeze(0)
        if meta_tensor.dim() == 1:
            meta_tensor = meta_tensor.unsqueeze(0)

        ecg_in = ecg_tensor.to(self.device)
        meta_in = meta_tensor.to(self.device)

        ecg_baseline = torch.zeros_like(ecg_in)
        meta_baseline = torch.zeros_like(meta_in)

        attr_ecg, _ = self.ig.attribute(
            inputs=(ecg_in, meta_in),
            baselines=(ecg_baseline, meta_baseline),
            target=target_class_idx,
            n_steps=n_steps,
            return_convergence_delta=False
        )

        attr_np = attr_ecg.squeeze(0).cpu().detach().numpy().T  # (1000, 12)

        # Spatial energy ranking across leads (L2 norm)
        lead_scores = np.linalg.norm(attr_np, axis=0)  # (12,)
        total_energy = np.sum(lead_scores) + 1e-8
        lead_importance = {
            LEAD_NAMES[i]: float(lead_scores[i] / total_energy)
            for i in range(len(LEAD_NAMES))
        }
        lead_importance = dict(sorted(lead_importance.items(), key=lambda item: item[1], reverse=True))

        return attr_np, lead_importance

    def explain_demographics_shap(self, sample_meta_batch, target_meta, target_class_idx, nsamples=40):
        """
        KernelSHAP attribution for Age and Sex.
        Returns:
            dict with SHAP impact values for Age and Sex.
        """
        def meta_predict_fn(meta_arr):
            with torch.no_grad():
                m_tensor = torch.tensor(meta_arr, dtype=torch.float32).to(self.device)
                dummy_ecg = torch.zeros((m_tensor.shape[0], 12, 1000), dtype=torch.float32).to(self.device)
                logits = self.model(dummy_ecg, m_tensor)
                probs = torch.sigmoid(logits)[:, target_class_idx].cpu().numpy()
            return probs

        background = sample_meta_batch[:min(40, len(sample_meta_batch))]
        explainer = shap.KernelExplainer(meta_predict_fn, background)
        shap_values = explainer.shap_values(target_meta.reshape(1, -1), nsamples=nsamples, silent=True)
        return {
            "Age": float(shap_values[0][0]),
            "Sex": float(shap_values[0][1])
        }