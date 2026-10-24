import os
import sys
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import glob
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, hamming_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import torch

from models.architectures import MODEL_REGISTRY, build_model
from training.train_module import LitCardioSignal
from xai.explainer import CLASSES, CLASS_DESCRIPTIONS

H5_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ptbxl_preprocessed.h5")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")


def get_model_checkpoints():
    """Maps model keys to their corresponding checkpoint file paths."""
    ckpt_map = {}
    for m_key in MODEL_REGISTRY:
        safe_slug = m_key.lower().replace("-", "_").replace(" ", "_")
        matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{safe_slug}*.ckpt"))
        if matches:
            ckpt_map[m_key] = matches[0]
        else:
            # Check for legacy cardiosignal naming
            if m_key == "CardioSignal-CrossAttn":
                legacy = glob.glob(os.path.join(CHECKPOINT_DIR, "cardiosignal_best*.ckpt"))
                if legacy:
                    ckpt_map[m_key] = legacy[0]
    return ckpt_map


def load_model_from_checkpoint(model_key, ckpt_path, device):
    """Loads model weights cleanly from PyTorch Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_key, num_classes=len(CLASSES)).to(device)
    
    state_dict = ckpt.get("state_dict", ckpt)
    clean_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=False)
    model.eval()
    return model


def benchmark_all_models():
    os.makedirs(REPORT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running CardioSignal-XAI Comprehensive Benchmark on Device: {device}")

    ckpt_map = get_model_checkpoints()
    print(f"[*] Found {len(ckpt_map)} trained model checkpoints in {CHECKPOINT_DIR}:")
    for k, v in ckpt_map.items():
        print(f"    - {k} -> {os.path.basename(v)}")

    if not ckpt_map:
        raise FileNotFoundError(f"No valid checkpoints found in {CHECKPOINT_DIR}")

    # Load Held-Out Test Set (Fold 10)
    print(f"[*] Loading Fold 10 Test Set from {H5_PATH}...")
    with h5py.File(H5_PATH, "r") as f:
        folds = f["folds"][:]
        test_indices = np.where(folds == 10)[0]
        y_true = f["labels"][test_indices]
        n_test = len(test_indices)
        print(f"[*] Test Set size: {n_test} patient ECG records")

        # Cache test tensors in memory for fast evaluation
        ecgs_test = torch.tensor(f["ecg_signals"][test_indices].transpose(0, 2, 1), dtype=torch.float32)
        metas_test = torch.tensor(f["metadata"][test_indices], dtype=torch.float32)

    all_model_results = {}
    table_rows = []
    roc_data = {}
    prc_data = {}
    calibration_data = {}

    batch_size = 128

    for m_key, ckpt_path in ckpt_map.items():
        print(f"\n{'='*55}")
        print(f"🔬 Benchmarking: {m_key}")
        print(f"{'='*55}")

        model = load_model_from_checkpoint(m_key, ckpt_path, device)
        total_params = sum(p.numel() for p in model.parameters())
        file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        # 1. Latency benchmark (100 sample warm-up and timed inference)
        sample_ecg = ecgs_test[:1].to(device)
        sample_meta = metas_test[:1].to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_ecg, sample_meta)
            
            t0 = time.time()
            n_iters = 50
            for _ in range(n_iters):
                _ = model(sample_ecg, sample_meta)
            latency_ms = ((time.time() - t0) / n_iters) * 1000.0

        # 2. Batch Inference on Test Set
        preds_list = []
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                b_ecg = ecgs_test[i:i + batch_size].to(device)
                b_meta = metas_test[i:i + batch_size].to(device)
                logits = model(b_ecg, b_meta)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds_list.append(probs)

        y_pred = np.vstack(preds_list)

        # 3. Compute Metrics
        # Per-class AUROC and AUPRC
        per_class_metrics = {}
        aurocs = []
        auprcs = []
        f1s = []
        sensitivities = []
        specificities = []
        brier_scores = []

        m_roc = {}
        m_prc = {}
        m_calib = {}

        for c_idx, cls_name in enumerate(CLASSES):
            yt = y_true[:, c_idx]
            yp = y_pred[:, c_idx]
            yp_binary = (yp >= 0.5).astype(int)

            fpr, tpr, _ = roc_curve(yt, yp)
            roc_auc = auc(fpr, tpr)
            aurocs.append(roc_auc)
            m_roc[cls_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}

            prec, rec, _ = precision_recall_curve(yt, yp)
            prc_auc = average_precision_score(yt, yp)
            auprcs.append(prc_auc)
            m_prc[cls_name] = {"precision": prec.tolist(), "recall": rec.tolist(), "auc": float(prc_auc)}

            # Calibration curve
            prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10)
            m_calib[cls_name] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}

            f1 = f1_score(yt, yp_binary, zero_division=0)
            rec_val = recall_score(yt, yp_binary, zero_division=0)
            prec_val = precision_score(yt, yp_binary, zero_division=0)
            
            # Specificity = TN / (TN + FP)
            tn = np.sum((yt == 0) & (yp_binary == 0))
            fp = np.sum((yt == 0) & (yp_binary == 1))
            spec = tn / (tn + fp + 1e-8)

            brier = brier_score_loss(yt, yp)

            f1s.append(f1)
            sensitivities.append(rec_val)
            specificities.append(spec)
            brier_scores.append(brier)

            per_class_metrics[cls_name] = {
                "AUROC": round(float(roc_auc), 4),
                "AUPRC": round(float(prc_auc), 4),
                "F1": round(float(f1), 4),
                "Precision": round(float(prec_val), 4),
                "Sensitivity": round(float(rec_val), 4),
                "Specificity": round(float(spec), 4),
                "BrierScore": round(float(brier), 4)
            }

        macro_auroc = float(np.mean(aurocs))
        macro_auprc = float(np.mean(auprcs))
        macro_f1 = float(np.mean(f1s))
        macro_sens = float(np.mean(sensitivities))
        macro_spec = float(np.mean(specificities))
        macro_brier = float(np.mean(brier_scores))

        y_pred_bin_all = (y_pred >= 0.5).astype(int)
        h_loss = float(hamming_loss(y_true, y_pred_bin_all))
        subset_acc = float(np.mean(np.all(y_true == y_pred_bin_all, axis=1)))

        all_model_results[m_key] = {
            "name": MODEL_REGISTRY[m_key]["name"],
            "family": MODEL_REGISTRY[m_key]["family"],
            "parameters": total_params,
            "checkpoint_size_mb": round(file_size_mb, 2),
            "latency_ms": round(latency_ms, 2),
            "macro_auroc": round(macro_auroc, 4),
            "macro_auprc": round(macro_auprc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_sensitivity": round(macro_sens, 4),
            "macro_specificity": round(macro_spec, 4),
            "brier_score": round(macro_brier, 4),
            "hamming_loss": round(h_loss, 4),
            "subset_accuracy": round(subset_acc, 4),
            "per_class": per_class_metrics
        }

        table_rows.append({
            "Model": m_key,
            "Family": MODEL_REGISTRY[m_key]["family"],
            "Macro AUROC": f"{macro_auroc:.4f}",
            "Macro AUPRC": f"{macro_auprc:.4f}",
            "Macro F1": f"{macro_f1:.4f}",
            "Sensitivity": f"{macro_sens:.4f}",
            "Specificity": f"{macro_spec:.4f}",
            "Brier Score": f"{macro_brier:.4f}",
            "Latency (ms)": f"{latency_ms:.2f}",
            "Params": f"{total_params:,}",
            "Size (MB)": f"{file_size_mb:.1f}"
        })

        roc_data[m_key] = m_roc
        prc_data[m_key] = m_prc
        calibration_data[m_key] = m_calib

    # Export CSV summary
    df_summary = pd.DataFrame(table_rows)
    df_summary.sort_values(by="Macro AUROC", ascending=False, inplace=True)
    csv_path = os.path.join(REPORT_DIR, "model_comparison_table.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[✓] Saved comparison table to: {csv_path}")

    # Export full JSON
    json_path = os.path.join(REPORT_DIR, "all_models_detailed_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_model_results, f, indent=2)
    print(f"[✓] Saved detailed metrics JSON to: {json_path}")

    # =================================================================
    # PLOT 1: Comparative Multi-Model ROC Curves (6 Subplots)
    # =================================================================
    colors = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261", "#9B5DE5", "#00B4D8"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for c_idx, cls_name in enumerate(CLASSES):
        ax = axes[c_idx]
        for m_idx, (m_key, m_rocs) in enumerate(roc_data.items()):
            c_roc = m_rocs[cls_name]
            ax.plot(
                c_roc["fpr"], c_roc["tpr"],
                label=f"{m_key} ({c_roc['auc']:.3f})",
                color=colors[m_idx % len(colors)],
                lw=2 if "CrossAttn" in m_key else 1.4
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        ax.set_title(f"Class: {cls_name} ({CLASS_DESCRIPTIONS[cls_name].split('(')[0].strip()})", fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    # Subplot 6: Macro Summary
    ax_macro = axes[5]
    for m_idx, m_key in enumerate(all_model_results):
        score = all_model_results[m_key]["macro_auroc"]
        ax_macro.barh(
            m_key, score,
            color=colors[m_idx % len(colors)],
            edgecolor="black",
            alpha=0.85
        )
        ax_macro.text(score - 0.08, m_key, f"{score:.4f}", va="center", ha="right", color="white", fontweight="bold", fontsize=9)
    ax_macro.set_xlim([0.5, 1.0])
    ax_macro.set_title("Overall Macro AUROC Leaderboard", fontsize=11, fontweight="bold")
    ax_macro.set_xlabel("Macro AUROC", fontsize=10)
    ax_macro.grid(alpha=0.3)

    plt.suptitle("CardioSignal-XAI: Cross-Model Multi-Label ROC Benchmark (PTB-XL Test Fold 10)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    roc_plot_path = os.path.join(REPORT_DIR, "model_comparison_roc.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"[✓] Saved high-resolution ROC figure to: {roc_plot_path}")

    # =================================================================
    # PLOT 2: Comparative Precision-Recall Curves
    # =================================================================
    fig_prc, axes_prc = plt.subplots(1, len(CLASSES), figsize=(20, 4.5))
    for c_idx, cls_name in enumerate(CLASSES):
        ax = axes_prc[c_idx]
        for m_idx, (m_key, m_prcs) in enumerate(prc_data.items()):
            c_prc = m_prcs[cls_name]
            ax.plot(
                c_prc["recall"], c_prc["precision"],
                label=f"{m_key} ({c_prc['auc']:.3f})",
                color=colors[m_idx % len(colors)],
                lw=1.8
            )
        ax.set_title(f"AUPRC: {cls_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    plt.suptitle("CardioSignal-XAI: Precision-Recall Curves by Diagnostic Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    prc_plot_path = os.path.join(REPORT_DIR, "model_comparison_prc.png")
    plt.savefig(prc_plot_path, dpi=300)
    plt.close()
    print(f"[✓] Saved high-resolution PRC figure to: {prc_plot_path}")

    # =================================================================
    # PLOT 3: Multi-Dimensional Radar / Spider Chart
    # =================================================================
    categories = ['AUROC', 'AUPRC', 'Macro F1', 'Sensitivity', 'Specificity', 'Efficiency']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig_radar = plt.figure(figsize=(9, 8))
    ax_radar = fig_radar.add_subplot(111, polar=True)

    max_lat = max(res["latency_ms"] for res in all_model_results.values()) + 1e-4

    for m_idx, (m_key, res) in enumerate(all_model_results.items()):
        # Efficiency = 1 - (latency / max_latency) scaled between 0.5 and 1.0
        eff = 0.5 + 0.5 * (1.0 - (res["latency_ms"] / max_lat))
        values = [
            res["macro_auroc"],
            res["macro_auprc"],
            res["macro_f1"],
            res["macro_sensitivity"],
            res["macro_specificity"],
            eff
        ]
        values += values[:1]
        ax_radar.plot(angles, values, lw=2, color=colors[m_idx % len(colors)], label=m_key)
        ax_radar.fill(angles, values, color=colors[m_idx % len(colors)], alpha=0.1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax_radar.set_ylim(0, 1.0)
    plt.title("CardioSignal-XAI: Multi-Model Holistic Performance & Efficiency Radar", size=13, fontweight="bold", y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
    radar_plot_path = os.path.join(REPORT_DIR, "model_radar_chart.png")
    plt.savefig(radar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved radar chart to: {radar_plot_path}")

    # =================================================================
    # PLOT 4: Calibration Reliability Curves
    # =================================================================
    fig_cal, axes_cal = plt.subplots(1, len(CLASSES), figsize=(20, 4.2))
    for c_idx, cls_name in enumerate(CLASSES):
        ax = axes_cal[c_idx]
        ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        for m_idx, (m_key, m_calibs) in enumerate(calibration_data.items()):
            c_cal = m_calibs[cls_name]
            ax.plot(
                c_cal["prob_pred"], c_cal["prob_true"], "s-",
                label=m_key,
                color=colors[m_idx % len(colors)],
                lw=1.5, markersize=4
            )
        ax.set_title(f"Reliability: {cls_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives", fontsize=10)
        ax.grid(alpha=0.3)
        if c_idx == 0:
            ax.legend(loc="upper left", fontsize=7)

    plt.suptitle("CardioSignal-XAI: Multi-Model Probability Calibration Diagrams", fontsize=13, fontweight="bold")
    plt.tight_layout()
    calib_plot_path = os.path.join(REPORT_DIR, "calibration_curves.png")
    plt.savefig(calib_plot_path, dpi=300)
    plt.close()
    print(f"[✓] Saved calibration diagram to: {calib_plot_path}")

    print("\n" + "="*60)
    print("🏆 FINAL BENCHMARK LEADERBOARD (FOLD 10)")
    print("="*60)
    print(df_summary.to_string(index=False))
    print("="*60)
    return all_model_results


if __name__ == "__main__":
    benchmark_all_models()
