import os
import glob
import torch
import numpy as np
import shap
from captum.attr import IntegratedGradients

from models.multimodal_model import CardioSignalXAI
from training.train_module import LitCardioSignal

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

class CardioExplainer:
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            ckpt_files = glob.glob(os.path.join("checkpoints", "*.ckpt"))
            if not ckpt_files:
                raise FileNotFoundError("No checkpoint found in checkpoints/")
            checkpoint_path = ckpt_files[0]

        print(f"[*] Loading model checkpoint from: {checkpoint_path}")
        lit_module = LitCardioSignal.load_from_checkpoint(checkpoint_path, map_location=self.device)
        self.model = lit_module.model.to(self.device)
        self.model.eval()

        # Integrated Gradients wrapper over (ECG, Metadata)
        self.ig = IntegratedGradients(self._forward_target)

    def _forward_target(self, ecg, metadata):
        return self.model(ecg, metadata)

    def predict(self, ecg_tensor, meta_tensor):
        """Returns class probabilities (5,)."""
        with torch.no_grad():
            ecg_in = ecg_tensor.unsqueeze(0).to(self.device)
            meta_in = meta_tensor.unsqueeze(0).to(self.device)
            logits = self.model(ecg_in, meta_in)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probs

    def explain_waveform_ig(self, ecg_tensor, meta_tensor, target_class_idx, n_steps=30):
        """
        Computes Integrated Gradients attribution for 12-lead ECG.
        ecg_tensor: (12, 1000)
        meta_tensor: (2,)
        Returns:
            attributions: (1000, 12) numpy array
            lead_importance: dict of {lead_name: score}
        """
        ecg_in = ecg_tensor.unsqueeze(0).to(self.device)
        meta_in = meta_tensor.unsqueeze(0).to(self.device)

        # Baseline: zero-amplitude flatline signal and neutral demographics
        ecg_baseline = torch.zeros_like(ecg_in)
        meta_baseline = torch.zeros_like(meta_in)

        attr_ecg, _ = self.ig.attribute(
            inputs=(ecg_in, meta_in),
            baselines=(ecg_baseline, meta_baseline),
            target=target_class_idx,
            n_steps=n_steps,
            return_convergence_delta=False
        )

        # Shape: (12, 1000) -> Transpose to (1000, 12)
        attr_np = attr_ecg.squeeze(0).cpu().detach().numpy().T

        # Novelty 3: Lead-wise L2 spatial energy ranking
        lead_scores = np.linalg.norm(attr_np, axis=0)  # (12,)
        total_energy = np.sum(lead_scores) + 1e-8
        lead_importance = {
            LEAD_NAMES[i]: float(lead_scores[i] / total_energy)
            for i in range(len(LEAD_NAMES))
        }

        # Sort leads descending by contribution
        lead_importance = dict(sorted(lead_importance.items(), key=lambda item: item[1], reverse=True))

        return attr_np, lead_importance

    def explain_demographics_shap(self, sample_meta_batch, target_meta, target_class_idx):
        """
        KernelSHAP attribution for Age and Sex.
        sample_meta_batch: (N, 2) background population
        target_meta: (2,) patient metadata
        """
        def meta_predict_fn(meta_arr):
            with torch.no_grad():
                m_tensor = torch.tensor(meta_arr, dtype=torch.float32).to(self.device)
                # Use zeroed synthetic ECG batch matching metadata sample count
                dummy_ecg = torch.zeros((m_tensor.shape[0], 12, 1000), dtype=torch.float32).to(self.device)
                logits = self.model(dummy_ecg, m_tensor)
                probs = torch.sigmoid(logits)[:, target_class_idx].cpu().numpy()
            return probs

        explainer = shap.KernelExplainer(meta_predict_fn, sample_meta_batch[:50])
        shap_values = explainer.shap_values(target_meta.reshape(1, -1), nsamples=50, silent=True)
        return {
            "Age": float(shap_values[0][0]),
            "Sex": float(shap_values[0][1])
        }