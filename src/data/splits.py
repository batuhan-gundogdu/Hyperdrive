import numpy as np
from collections import defaultdict

import config


def split_by_patient(entries, train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO,
                     seed=config.RANDOM_SEED):
    """Split entries into train/val/test sets at the patient level."""
    # Group entries by subject_id
    patient_entries = defaultdict(list)
    for entry in entries:
        patient_entries[entry["subject_id"]].append(entry)

    patient_ids = sorted(patient_entries.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train:n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val:])

    train_entries = [e for e in entries if e["subject_id"] in train_ids]
    val_entries = [e for e in entries if e["subject_id"] in val_ids]
    test_entries = [e for e in entries if e["subject_id"] in test_ids]

    return train_entries, val_entries, test_entries


def get_finetune_candidates(test_entries, min_recordings=config.MIN_RECORDINGS_FOR_FINETUNE):
    """Find test patients with enough recordings and both normal+abnormal labels."""
    patient_data = defaultdict(lambda: {"normal": 0, "abnormal": 0, "total": 0})
    for entry in test_entries:
        pid = entry["subject_id"]
        patient_data[pid]["total"] += 1
        if entry["is_normal"]:
            patient_data[pid]["normal"] += 1
        else:
            patient_data[pid]["abnormal"] += 1

    candidates = []
    for pid, counts in patient_data.items():
        if counts["total"] >= min_recordings and counts["normal"] > 0 and counts["abnormal"] > 0:
            candidates.append(pid)

    return sorted(candidates)
