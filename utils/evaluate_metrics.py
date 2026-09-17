import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import torch

from training.train_module import LitCardioSignal
from xai.explainer import CLASSES

H5_PATH = "data/processed/ptbxl_preprocessed.h5"
REPORT_DIR = "reports"

def generate_evaluation_artifacts():
    os.makedirs(REPORT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_files = glob.glob(os.path.join("checkpoints", "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint found.")
    
    print(f"[*] Loading model from {ckpt_files[0]}...")
    lit_model = LitCardioSignal.load_from_checkpoint(ckpt_files[0], map_location=device)
    model = lit_model.model.to(device)
    model.eval()

    print("[*] Running inference across entire Fold 10 test set...")
    with h5py.File(H5_PATH, "r") as f:
        folds = f["folds"][:]
        test_indices = np.where(folds == 10)[0]
        
        all_preds = []
        all_targets = []
        
        # Batch evaluation
        batch_size = 64
        for i in range(0, len(test_indices), batch_size):
            batch_idx = test_indices[i:i + batch_size]
            ecgs = torch.tensor(f["ecg_signals"][batch_idx].transpose(0, 2, 1), dtype=torch.float32).to(device)
            metas = torch.tensor(f["metadata"][batch_idx], dtype=torch.float32).to(device)
            labels = f["labels"][batch_idx]

            with torch.no_grad():
                probs = torch.sigmoid(model(ecgs, metas)).cpu().numpy()
                all_preds.append(probs)
                all_targets.append(labels)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # 1. Plot Multilabel ROC Curves
    plt.figure(figsize=(8, 6))
    metrics_summary = []

    for i, cls_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        pr_auc = average_precision_score(y_true[:, i], y_pred[:, i])

        plt.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUROC = {roc_auc:.3f})")
        metrics_summary.append({
            "Disease Class": cls_name,
            "AUROC": f"{roc_auc:.4f}",
            "AUPRC": f"{pr_auc:.4f}"
        })

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("CardioSignal-XAI: Multi-Label ROC Curves (PTB-XL Test Fold 10)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "roc_curves.png"), dpi=300)
    plt.close()

    # 2. Output Quantitative Summary
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(REPORT_DIR, "test_metrics_table.csv"), index=False)
    
    print("\n" + "="*45)
    print("FINAL TEST PERFORMANCE (FOLD 10)")
    print("="*45)
    print(df_metrics.to_string(index=False))
    print(f"\n[✓] Saved high-res ROC plot to: {REPORT_DIR}/roc_curves.png")
    print(f"[✓] Saved metrics table to: {REPORT_DIR}/test_metrics_table.csv")

if __name__ == "__main__":
    generate_evaluation_artifacts()