# MIMIC-IV-ECG v1.0 — Data Structure Analysis

## Directory Hierarchy

```
physionet.org/
├── files/mimic-iv-ecg/1.0/
│   ├── record_list.csv                          # Maps subject_id → study_id → file path
│   ├── machine_measurements.csv                 # Machine-extracted features + text reports
│   ├── machine_measurements_data_dictionary.csv
│   ├── files/
│   │   ├── p1000/                               # Patient group prefix (first 4 digits)
│   │   │   ├── p10000032/                       # Full patient ID (subject_id)
│   │   │   │   ├── s40689238/                   # Study ID
│   │   │   │   │   ├── 40689238.hea             # WFDB header (metadata)
│   │   │   │   │   └── 40689238.dat             # WFDB signal data (raw waveform)
│   │   │   │   ├── s44458630/
│   │   │   │   ...
```

## Key Numbers

| Metric                            | Value                              |
| --------------------------------- | ---------------------------------- |
| Unique patients (`subject_id`)    | 161,352                            |
| Total ECG recordings              | 800,035                            |
| Studies per patient               | min: 1, median: 2, mean: ~5, max: 260 |
| Patient group folders (`pXXXX`)   | 1,001                              |

## Signal Format (WFDB)

Each ECG recording has a consistent format:

- **12 leads**: I, II, III, aVR, aVF, aVL, V1–V6
- **Sampling rate**: 500 Hz
- **Duration**: 5,000 samples = 10 seconds
- **Resolution**: 16-bit, 200 ADC units/mV
- **File size**: ~120 KB per `.dat` file

### Example Header (`.hea`)

```
45106466 12 500 5000 12:04:00 22/09/2182
45106466.dat 16 200.0(0)/mV 16 0 4 14670 0 I
45106466.dat 16 200.0(0)/mV 16 0 4 4792 0 II
45106466.dat 16 200.0(0)/mV 16 0 -2 45676 0 III
45106466.dat 16 200.0(0)/mV 16 0 -4 55945 0 aVR
45106466.dat 16 200.0(0)/mV 16 0 1 57969 0 aVF
45106466.dat 16 200.0(0)/mV 16 0 4 22036 0 aVL
45106466.dat 16 200.0(0)/mV 16 0 4 32604 0 V1
45106466.dat 16 200.0(0)/mV 16 0 9 19253 0 V2
45106466.dat 16 200.0(0)/mV 16 0 7 29742 0 V3
45106466.dat 16 200.0(0)/mV 16 0 2 45493 0 V4
45106466.dat 16 200.0(0)/mV 16 0 5 63527 0 V5
45106466.dat 16 200.0(0)/mV 16 0 2 3458 0 V6
# <subject_id>: 12177274
```

## Machine Measurements

Available features in `machine_measurements.csv` (per study):

| Feature       | Description                                  | Unit    |
| ------------- | -------------------------------------------- | ------- |
| `rr_interval` | Time between successive R-waves              | msec    |
| `p_onset`     | Time at the onset of the P-wave              | msec    |
| `p_end`       | Time at the end of the P-wave                | msec    |
| `qrs_onset`   | Time at the beginning of the QRS complex     | msec    |
| `qrs_end`     | Time at the end of the QRS complex           | msec    |
| `t_end`       | Time at the end of the T-wave                | msec    |
| `p_axis`      | Electrical axis of the P-wave                | degrees |
| `qrs_axis`    | Electrical axis of the QRS complex           | degrees |
| `t_axis`      | Electrical axis of the T-wave                | degrees |

Additionally, up to 18 free-text report fields (`report_0` through `report_17`) contain machine-generated cardiology interpretations (e.g., "Sinus rhythm", "Normal ECG", "Sinus tachycardia").

## Metadata CSVs

### `record_list.csv`

Maps each recording to its file path:

```
subject_id, study_id, file_name, ecg_time, path
10000032,   40689238, 40689238,  2180-07-23 08:44:00, files/p1000/p10000032/s40689238/40689238
```

### `machine_measurements.csv`

Contains extracted features and text reports per study, keyed by `subject_id` and `study_id`.

## Per-Patient Anomaly Detection Considerations

- Each patient has a **time-ordered series** of ECG studies (via `ecg_time`), enabling temporal modeling.
- **Input options**:
  - Raw 12-lead waveforms: 5,000 samples × 12 leads = 60,000 values per recording.
  - Machine measurements: 9 numeric features per recording.
  - Combined: both waveform and measurement features.
- With a median of 2 studies per patient (mean ~5), patients with more recordings provide more training data for personalized models.
- The text reports can serve as weak labels for defining "normal" vs. "anomalous" patterns.
