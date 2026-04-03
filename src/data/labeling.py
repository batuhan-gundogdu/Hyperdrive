import pandas as pd
import config


NORMAL_KEYWORDS = ["Normal ECG", "Sinus rhythm"]


def is_normal_report(row):
    """Determine if an ECG recording is normal based on machine-generated reports."""
    for i in range(18):
        col = f"report_{i}"
        if col not in row.index:
            break
        val = row[col]
        if pd.isna(val):
            continue
        val = str(val).strip()
        for keyword in NORMAL_KEYWORDS:
            if val.startswith(keyword):
                return True
    return False


def build_label_map():
    """Build a mapping from (subject_id, study_id) -> is_normal."""
    df = pd.read_csv(config.MEASUREMENTS_CSV)
    # Vectorized: check if any report column starts with a normal keyword
    report_cols = [c for c in df.columns if c.startswith("report_")]
    is_normal = pd.Series(False, index=df.index)
    for col in report_cols:
        col_str = df[col].astype(str).str.strip()
        for keyword in NORMAL_KEYWORDS:
            is_normal = is_normal | col_str.str.startswith(keyword, na=False)
    labels = {}
    for sid, study, normal in zip(df["subject_id"], df["study_id"], is_normal):
        labels[(int(sid), int(study))] = bool(normal)
    return labels


def build_record_list():
    """Build the full record list with paths and labels."""
    records = pd.read_csv(config.RECORD_LIST_CSV)
    labels = build_label_map()

    # Vectorized: build entries using pandas operations
    records["_key"] = list(zip(records["subject_id"].astype(int), records["study_id"].astype(int)))
    records["is_normal"] = records["_key"].map(labels)
    records = records.dropna(subset=["is_normal"])

    entries = []
    for subject_id, study_id, path, ecg_time, is_normal in zip(
        records["subject_id"], records["study_id"], records["path"],
        records["ecg_time"], records["is_normal"]
    ):
        entries.append({
            "subject_id": int(subject_id),
            "study_id": int(study_id),
            "dat_path": f"{config.DATA_ROOT}/{path}.dat",
            "ecg_time": ecg_time,
            "is_normal": bool(is_normal),
        })
    return entries
