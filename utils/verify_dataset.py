import os
import pandas as pd
import wfdb

DATA_DIR = os.path.join("data", "raw", "ptb-xl-1.0.3")

def test_data():
    db_csv = os.path.join(DATA_DIR, "ptbxl_database.csv")
    scp_csv = os.path.join(DATA_DIR, "scp_statements.csv")
    
    if not os.path.exists(db_csv) or not os.path.exists(scp_csv):
        print(f"[!] Missing CSV files in {DATA_DIR}")
        return
    
    df = pd.read_csv(db_csv, index_col="ecg_id")
    print(f"[✓] Metadata CSV loaded: {df.shape[0]} patient records found.")
    
    first_record_rel = df.iloc[0]["filename_lr"]
    record_path = os.path.join(DATA_DIR, first_record_rel)
    
    signal, meta = wfdb.rdsamp(record_path)
    print(f"[✓] Sample ECG read: Shape = {signal.shape} (1000 timepoints, 12 leads)")
    print(f"[✓] Sampling frequency: {meta['fs']} Hz")
    print(f"[✓] Leads: {meta['sig_name']}")

if __name__ == "__main__":
    test_data()