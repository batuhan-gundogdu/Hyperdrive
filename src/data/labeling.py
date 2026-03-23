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
    labels = {}
    for _, row in df.iterrows():
        key = (int(row["subject_id"]), int(row["study_id"]))
        labels[key] = is_normal_report(row)
    return labels


def build_record_list():
    """Build the full record list with paths and labels."""
    records = pd.read_csv(config.RECORD_LIST_CSV)
    labels = build_label_map()

    entries = []
    for _, row in records.iterrows():
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        dat_path = f"{config.DATA_ROOT}/{row['path']}.dat"
        is_normal = labels.get((subject_id, study_id), None)
        if is_normal is None:
            continue
        entries.append({
            "subject_id": subject_id,
            "study_id": study_id,
            "dat_path": dat_path,
            "ecg_time": row["ecg_time"],
            "is_normal": is_normal,
        })
    return entries
