import os
import ast
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import wfdb

RAW_DATA_DIR = os.path.join("data", "raw", "ptb-xl-1.0.3")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_H5 = os.path.join(OUTPUT_DIR, "ptbxl_preprocessed.h5")

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLING_RATE = 100
SIGNAL_LEN = 1000
NUM_LEADS = 12

def create_butter_bandpass(lowcut=0.5, highcut=45.0, fs=100.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")

def apply_bandpass_filter(signal, b, a):
    return filtfilt(b, a, signal, axis=0)

def standardize_signal(signal, eps=1e-8):
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    return (signal - mean) / (std + eps)

def aggregate_diagnostic_superclasses(df, scp_statements):
    diag_scp = scp_statements[scp_statements.diagnostic == 1]
    code_to_superclass = {
        code: row.diagnostic_class
        for code, row in diag_scp.iterrows()
        if pd.notna(row.diagnostic_class)
    }

    def get_labels(scp_dict_str):
        labels = set()
        scp_dict = ast.literal_eval(scp_dict_str)
        for code in scp_dict.keys():
            if code in code_to_superclass:
                sclass = code_to_superclass[code]
                if sclass in CLASSES:
                    labels.add(sclass)
        return list(labels)

    df["diagnostic_superclasses"] = df.scp_codes.apply(get_labels)
    
    labels_matrix = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for idx, row_labels in enumerate(df["diagnostic_superclasses"]):
        for label in row_labels:
            class_idx = CLASSES.index(label)
            labels_matrix[idx, class_idx] = 1.0

    return df, labels_matrix

def prepare_metadata(df):
    sex_numeric = df["sex"].map({"Male": 1.0, "Female": 0.0, 1: 1.0, 0: 0.0})
    sex_numeric = sex_numeric.fillna(sex_numeric.mode()[0]).values

    age_numeric = df["age"].copy()
    age_median = age_numeric.median()
    age_numeric = age_numeric.fillna(age_median).values.reshape(-1, 1)

    scaler = StandardScaler()
    age_scaled = scaler.fit_transform(age_numeric).flatten()

    return np.stack([age_scaled, sex_numeric], axis=1).astype(np.float32)

def build_hdf5_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    db_path = os.path.join(RAW_DATA_DIR, "ptbxl_database.csv")
    scp_path = os.path.join(RAW_DATA_DIR, "scp_statements.csv")

    if not os.path.exists(db_path) or not os.path.exists(scp_path):
        raise FileNotFoundError(f"Missing required CSVs in {RAW_DATA_DIR}")

    print("[*] Loading metadata CSVs...")
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)

    print("[*] Aggregating multi-label diagnostic targets...")
    df, labels_matrix = aggregate_diagnostic_superclasses(df, scp_df)

    print("[*] Preprocessing Age & Sex metadata...")
    metadata = prepare_metadata(df)

    b, a = create_butter_bandpass(lowcut=0.5, highcut=45.0, fs=SAMPLING_RATE, order=3)
    num_samples = len(df)
    folds = df["strat_fold"].values.astype(np.int32)
    ecg_ids = df.index.values.astype(np.int32)

    print(f"[*] Building HDF5 tensor file: {OUTPUT_H5}")
    with h5py.File(OUTPUT_H5, "w") as h5_file:
        ecg_dataset = h5_file.create_dataset(
            "ecg_signals",
            shape=(num_samples, SIGNAL_LEN, NUM_LEADS),
            dtype=np.float32,
            chunks=(64, SIGNAL_LEN, NUM_LEADS),
            compression="gzip"
        )
        h5_file.create_dataset("metadata", data=metadata, dtype=np.float32)
        h5_file.create_dataset("labels", data=labels_matrix, dtype=np.float32)
        h5_file.create_dataset("folds", data=folds, dtype=np.int32)
        h5_file.create_dataset("ecg_ids", data=ecg_ids, dtype=np.int32)

        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=num_samples, desc="Filtering signals")):
            rel_path = row["filename_lr"]
            full_record_path = os.path.join(RAW_DATA_DIR, rel_path)

            try:
                raw_sig, _ = wfdb.rdsamp(full_record_path)
                if raw_sig.shape[0] < SIGNAL_LEN:
                    pad_len = SIGNAL_LEN - raw_sig.shape[0]
                    raw_sig = np.pad(raw_sig, ((0, pad_len), (0, 0)), mode="edge")
                elif raw_sig.shape[0] > SIGNAL_LEN:
                    raw_sig = raw_sig[:SIGNAL_LEN, :]

                filtered_sig = apply_bandpass_filter(raw_sig, b, a)
                normalized_sig = standardize_signal(filtered_sig)
                ecg_dataset[i] = normalized_sig.astype(np.float32)
            except Exception as e:
                print(f"[!] Warning on record {full_record_path}: {e}")
                ecg_dataset[i] = np.zeros((SIGNAL_LEN, NUM_LEADS), dtype=np.float32)

        h5_file.attrs["classes"] = [c.encode("utf-8") for c in CLASSES]

    print("\n[✓] Preprocessing complete. HDF5 saved to:", OUTPUT_H5)

if __name__ == "__main__":
    build_hdf5_dataset()