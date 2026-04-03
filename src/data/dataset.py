import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_lead_ii_windows


class ECGWindowDataset(Dataset):
    """Dataset that returns individual 1-second ECG windows."""

    def __init__(self, entries, global_mean=0.0, global_std=1.0, windows_per_sample=1):
        """
        Args:
            entries: list of dicts with keys: subject_id, study_id, dat_path, is_normal
            global_mean: mean for z-score normalization
            global_std: std for z-score normalization
            windows_per_sample: number of random windows to return per __getitem__ call
                                (1 for training, 10 for evaluation to get all windows)
        """
        self.entries = entries
        self.global_mean = global_mean
        self.global_std = global_std
        self.windows_per_sample = windows_per_sample

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        windows = load_lead_ii_windows(entry["dat_path"])
        windows = (windows - self.global_mean) / (self.global_std + 1e-8)
        windows = np.clip(windows, -5.0, 5.0)  # Clip to ±5 std to handle corrupted recordings

        if self.windows_per_sample == 1:
            # Random window for training
            win_idx = np.random.randint(0, windows.shape[0])
            signal = torch.tensor(windows[win_idx], dtype=torch.float32)
        else:
            # All windows for evaluation
            signal = torch.tensor(windows, dtype=torch.float32)

        return {
            "signal": signal,
            "subject_id": entry["subject_id"],
            "study_id": entry["study_id"],
            "label": int(not entry["is_normal"]),  # 1 = abnormal, 0 = normal
        }


class PatientDataset(ECGWindowDataset):
    """Dataset filtered to a single patient's recordings."""

    def __init__(self, entries, subject_id, global_mean=0.0, global_std=1.0,
                 normal_only=True, windows_per_sample=1):
        patient_entries = [e for e in entries if e["subject_id"] == subject_id]
        if normal_only:
            patient_entries = [e for e in patient_entries if e["is_normal"]]
        super().__init__(patient_entries, global_mean, global_std, windows_per_sample)
        self.subject_id = subject_id
