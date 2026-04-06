import os
import ast
import numpy as np
import pandas as pd
import wfdb

DATA_DIR = "ptbxl"
OUT_DIR = "data/processed_cv"

SAMPLING_RATE = 500
INPUT_LENGTH = 5000
NUM_LEADS = 12

os.makedirs(OUT_DIR, exist_ok=True)


def parse_scp_codes(s):
    return ast.literal_eval(s)


def load_raw_data(df, sampling_rate, path):
    signals = []
    ids = []

    for _, row in df.iterrows():
        f = row["filename_hr"] if sampling_rate == 500 else row["filename_lr"]
        signal, _ = wfdb.rdsamp(os.path.join(path, f))
        signals.append(signal.astype(np.float32))
        ids.append(int(row["ecg_id"]))

    return np.array(signals, dtype=np.float32), np.array(ids, dtype=np.int64)


def is_mi_record(scp_codes, scp_df):
    for code in scp_codes.keys():
        if code in scp_df.index:
            row = scp_df.loc[code]
            diagnostic_flag = row.get("diagnostic", 0.0)
            diagnostic_class = row.get("diagnostic_class", "")
            if diagnostic_flag == 1.0 and diagnostic_class == "MI":
                return True
    return False


def is_clean_normal_record(scp_codes):
    keys = set(scp_codes.keys())
    return keys == {"NORM"}


def normalize_multilead(signal):
    out = np.zeros_like(signal, dtype=np.float32)
    for c in range(signal.shape[0]):
        x = signal[c].astype(np.float32)
        x = x - np.mean(x)
        std = np.std(x)
        if std > 1e-6:
            x = x / std
        out[c] = x
    return out


def process_signal(signal):
    # input [T, 12] -> output [12, 5000]
    if signal.shape[0] > INPUT_LENGTH:
        signal = signal[:INPUT_LENGTH, :]
    elif signal.shape[0] < INPUT_LENGTH:
        padded = np.zeros((INPUT_LENGTH, signal.shape[1]), dtype=np.float32)
        padded[:signal.shape[0], :] = signal
        signal = padded

    signal = signal.T
    signal = normalize_multilead(signal)
    return signal.astype(np.float32)


def main():
    print("Loading metadata...")
    db = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"))
    db["scp_codes"] = db["scp_codes"].apply(parse_scp_codes)

    scp_df = pd.read_csv(os.path.join(DATA_DIR, "scp_statements.csv"), index_col=0)

    selected_rows = []
    labels = []

    print("Filtering strict NORM vs MI...")
    for _, row in db.iterrows():
        scp_codes = row["scp_codes"]
        mi_flag = is_mi_record(scp_codes, scp_df)
        norm_flag = is_clean_normal_record(scp_codes)

        if mi_flag:
            selected_rows.append(row)
            labels.append(1)
        elif norm_flag:
            selected_rows.append(row)
            labels.append(0)

    filt_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    filt_df["label"] = labels

    print("\nClass counts before any split:")
    print(filt_df["label"].value_counts())

    print("\nLoading 500 Hz waveforms...")
    X_all, ecg_ids = load_raw_data(filt_df, SAMPLING_RATE, DATA_DIR)

    X = []
    y = []
    ids = []
    folds = []

    for i in range(len(filt_df)):
        signal = X_all[i]  # [5000, 12]
        if signal.ndim != 2 or signal.shape[1] != NUM_LEADS:
            continue

        processed = process_signal(signal)  # [12, 5000]

        X.append(processed)
        y.append(int(filt_df.loc[i, "label"]))
        ids.append(int(filt_df.loc[i, "ecg_id"]))
        folds.append(int(filt_df.loc[i, "strat_fold"]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    ids = np.array(ids, dtype=np.int64)
    folds = np.array(folds, dtype=np.int64)

    print("\nFinal X shape:", X.shape)
    print("Final y shape:", y.shape)

    for f in range(1, 11):
        mask = folds == f
        print(f"Fold {f}: count={mask.sum()}, class_counts={np.bincount(y[mask]) if mask.sum() > 0 else 'empty'}")

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "ids.npy"), ids)
    np.save(os.path.join(OUT_DIR, "folds.npy"), folds)

    print("\nSaved CV dataset.")


if __name__ == "__main__":
    main()
